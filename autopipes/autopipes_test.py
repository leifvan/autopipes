from multiprocessing import Process
from time import sleep

from .autopipes import Pipeline
from .visualize import visualize_pipeline


def make_util_process(target, *args):
    process = Process(target=target, args=args, daemon=True)
    process.start()
    return process


def source_thread(in_queues, out_queues):
    for i in range(100):
        out_queues["source_rgbd"].put(f"frame {i}")
        sleep(0.1)
        #print("emitted source frame", i, flush=True)


def source_fn(in_queues, out_queues, events):
    make_util_process(source_thread, in_queues, out_queues)


def seg_thread(in_queues, out_queues):
    while True:
        rgbd = in_queues["source_rgbd"].get()
        out_queues["segmentation_result"].put("segmented " + rgbd)
        #print("segmented", rgbd, flush=True)


def seg_fn(in_queues, out_queues, events):
    make_util_process(seg_thread, in_queues, out_queues)


def separation_thread(in_queues, out_queues):
    while True:
        rgbd = in_queues["source_rgbd"].get()
        segmentation = in_queues["postprocessed_segmentation_result"].get()
        out_queues["static_rgbd"].put("static part of " + segmentation)
        out_queues["dynamic_rgbd"].put("dynamic part of " + rgbd)
        # print("separated", rgbd, "using", segmentation, flush=True)


def sep_fn(in_queues, out_queues, events):
    make_util_process(separation_thread, in_queues, out_queues)


def server_thread(in_queues, out_queues):
    while True:
        static_rgbd = in_queues["static_rgbd"].get()
        dynamic_rgbd = in_queues["dynamic_rgbd"].get()
        # print("got static:", static_rgbd, flush=True)
        # print("got dynamic:", dynamic_rgbd, flush=True)


def server_fn(in_queues, out_queues, events):
    make_util_process(server_thread, in_queues, out_queues)


if __name__ == '__main__':
    pipeline = Pipeline(debug_flow_queues='all')

    pipeline.add_node(
        name="source",
        init_fn=source_fn,
        outlet_names=("source_rgbd",)
    )
    pipeline.add_node(
        name="segmentation",
        init_fn=seg_fn,
        inlet_names=("source_rgbd",),
        outlet_names=("segmentation_result",)
    )

    pipeline.add_optional_bridge("segmentation_result", "postprocessed_segmentation_result")

    pipeline.add_node(
        name="separation",
        init_fn=sep_fn,
        inlet_names=("postprocessed_segmentation_result", "source_rgbd"),
        outlet_names=("static_rgbd", "dynamic_rgbd", "empty_outlet")
    )

    pipeline.add_node(
        name="server",
        init_fn=server_fn,
        inlet_names=("static_rgbd", "dynamic_rgbd")
    )

    pipeline.initialize(auto_create_sinks=True)
    visualize_pipeline(pipeline)

    while True:
        pass
