import network

networks = []
errors = []

def build_networks(layers, activations, d_activations, cost, d_cost, num_nets,
        random_limit):
    for i in range(num_nets):
        networks.append(network.FeedForwardNetwork(layers, activations,
            d_activations, cost, d_cost, random_limit))
        errors.append(0)

def train_nets(inputs, outputs, training_rate, epochs, batch_size, outer_min,
        random_limit, layers, activations, d_activations, cost, d_cost, num_nets):
    minimum = 100
    minnet = 3
    while (minimum > outer_min): #or random_limit>100001):
        sum = 0
        #while minimum > outer_min:
        print('building networks')
        build_networks(layers, activations, d_activations, cost, d_cost,
                num_nets, random_limit)
        for network in networks:
            output = network.train(inputs, outputs, training_rate, epochs,
                    batch_size, False)
            print('finished a network')
            print("output error =", output)
            # Everything below subjected to changes
            sum += output
            if output < minimum:
                minimum = output
                minnet = network
            if minimum < outer_min:
                return minnet
        print("Couldn't find anything")
        avg = sum / num_nets
        print(minimum, avg, random_limit)
        random_limit *= 10
    return minnet
