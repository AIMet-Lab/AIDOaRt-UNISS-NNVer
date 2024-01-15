import pynever.nodes as pyn_nodes
import pynever.networks as pyn_networks


def is_convolutional(pyn_net: pyn_networks.SequentialNetwork) -> bool:

    current_node = pyn_net.get_first_node()
    is_conv = False
    while current_node is not None:

        if isinstance(current_node, pyn_nodes.ConvNode):
            is_conv = True

        current_node = pyn_net.get_next_node(current_node)

    return is_conv


def extract_cls(pyn_net: pyn_networks.SequentialNetwork) -> pyn_networks.SequentialNetwork:

    pyn_cls = pyn_networks.SequentialNetwork(pyn_net.identifier + "_CLS", "X")
    current_node = pyn_net.get_first_node()
    flatten_reached = False
    while current_node is not None:

        # We scroll the network until we reach the flatten layer
        if not flatten_reached:

            if isinstance(current_node, pyn_nodes.FlattenNode):
                flatten_reached = True

        # Then we add all the following nodes to the network to verify
        else:
            pyn_cls.add_node(current_node)

        current_node = pyn_net.get_next_node(current_node)

    return pyn_cls

