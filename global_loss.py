import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Import tensornetwork
import tensornetwork as tn
# Set the backend to tesorflow
# (default is numpy)
tn.set_default_backend("tensorflow")

with tf.device('/gpu:3'):
    class TNLayer(tf.keras.layers.Layer):

        def __init__(self, num_node, bond_dim, physical_dim):
            super(TNLayer, self).__init__()
            self.num_node = num_node
            self.bond_dim = bond_dim
            self.physical_dim = physical_dim
            self.vars = tf.constant(tf.random.uniform(shape=(num_node, bond_dim, physical_dim, bond_dim))-0.5)
            #self.contant = tf.constant(tf.random.uniform(shape=(num_node, bond_dim, physical_dim, bond_dim))-1)
        def call(self, constant):
            def build_block(num_node, bond_dim, physical_dim, variables):
                a_list = []

                variables = tf.constant(variables)
                a = tn.Node(variables[0])
                a_list.append(a)
                for i in range(num_node-1):
                    a =tn.Node(variables[i+1])
                    a_list[-1][-1]^ a[0]
                    a_list.append(a)

                a_list[-1][-1]^a_list[0][0]
                node = a_list[0]
                for i in range(num_node-1):
                    node = tn.contract_between(node,a_list[i+1])

                return node
            def overlap(node1, node2):
                assert len(node1.tensor.shape) == len(node2.tensor.shape)
                num = len(node1.tensor.shape)

                for i in range(num):
                    node1[i]^node2[i]
                return tn.contract_between(node1,node2).tensor


            def loss_function(variables, constant):
                num_node, bond_dim, physical_dim = variables.shape[:3]

                nodes1 = build_block(num_node, bond_dim, physical_dim, variables)
                nodes2 = build_block(num_node, bond_dim, physical_dim, constant)
                nodes_norm1 = build_block(num_node, bond_dim, physical_dim, variables)
                nodes_norm2 = build_block(num_node, bond_dim, physical_dim, variables)
                nodes2_norm1 = build_block(num_node, bond_dim, physical_dim, constant)
                nodes2_norm2 = build_block(num_node, bond_dim, physical_dim, constant)
                norm = overlap(nodes_norm1, nodes_norm2)
                norm2 = overlap(nodes2_norm1, nodes2_norm2)
                output = overlap(nodes1, nodes2)/tf.math.sqrt(norm*norm2)
                return output
            

            loss = loss_function(self.vars, constant)
            return loss

    bond_dim = 2
    physical_dim = 2

    file = open("global_step.txt","w")
    file.close()
    output_list = []
    #constant = tf.constant(tf.random.uniform(shape=(num_node, bond_dim, physical_dim, bond_dim))-0.5)
    for num_node in range(5,21):
        file = open("global_step.txt","a")
        print("num_node is {0}".format(num_node))
        do_dx_list = []
        constant = tf.constant(tf.ones((num_node, bond_dim, physical_dim, bond_dim))-0.5)  
        mean_result = 10**-34
        for i in range(1000000):
        
            #for i in range(1):#test
            #print("i is {0}".format(i))
            layer1 = TNLayer(num_node, bond_dim, physical_dim)
             

            with tf.GradientTape(persistent=True) as t:
                t.watch(layer1.vars)
                output = layer1(constant)
            do_dx = 2*output*t.gradient(output, layer1.vars)
            #print(do_dx[0,0,0,0])
            do_dx_list.append(do_dx)
            if i %10000 == 0 and np.abs(np.mean(np.array(do_dx_list)**2)-mean_result)/mean_result<10**-3:
                print("mean value is {0} and break".format(mean_result)) 
                break
            elif i%10000 == 0:
                print("variance value is {0}".format(mean_result))
                print("mean value is {0}".format(np.mean(np.array(do_dx_list))))
                print("number of values is {0}".format(len(do_dx_list)))
                mean_result = np.mean(np.array(do_dx_list)**2)
        output_list.append(np.mean(np.array(do_dx_list)**2))
        file.write(str(output_list[-1]))
        file.write("\n")
        file.close()
    np.savetxt("global",output_list)





