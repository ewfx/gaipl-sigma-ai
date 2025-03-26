import graphviz

def generate_dot(app_name, network_data, api_filter=None):
    dot = graphviz.Digraph(comment=f'Full Network View for {app_name}')
    dot.attr(rankdir='LR', size='8,5')

    dot.node(app_name, shape="box", style="filled", color="lightblue")

    # Show outbound connections
    app_entry = next((item for item in network_data if item["app"] == app_name), None)
    if app_entry:
        for api in app_entry["api_flows"]:
            target = api["connects_to"]
            label = api["api"]
            if api_filter and api_filter != label:
                continue
            dot.node(target, shape="ellipse", style="filled", color="lightgray")
            dot.edge(app_name, target, label=label)

    # Show inbound connections (other apps connecting to this one)
    for other in network_data:
        if other["app"] == app_name:
            continue
        for api in other["api_flows"]:
            if api["connects_to"] == app_name:
                label = api["api"]
                if api_filter and api_filter != label:
                    continue
                dot.node(other["app"], shape="ellipse", style="filled", color="lightgray")
                dot.edge(other["app"], app_name, label=label)

    return dot.source