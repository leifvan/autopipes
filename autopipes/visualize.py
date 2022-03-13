import graphviz

from autopipes import Pipeline


def visualize_pipeline(pipeline: Pipeline, output_path="temp"):
    dot = graphviz.Digraph()

    for node_name in pipeline.nodes.keys():
        dot.node(node_name, label="=>" if "=>" in node_name else node_name)

    pipes = pipeline._get_pipes_dict()
    for name, pipe in pipes.items():
        if pipe.inlet_node is None:
            print("pipe", name, "has no inlet")
        else:
            for outlet in pipe.outlet_nodes:
                dot.edge(pipe.inlet_node.name, outlet.name, label=name)

    dot.render(output_path, view=True)
