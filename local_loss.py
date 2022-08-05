import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# Import tensornetwork
import tensornetwork as tn
# Set the backend to tesorflow
# (default is numpy)
tn.set_default_backend("tensorflow")
bond_dim = 2
physical_dim = 2



import copy


with tf.device('/gpu:1'):


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
                a = tn.Node(variables[0][0,:,:])
                a_list.append(a)
                for i in range(num_node-1):
                    if i+1 == num_node-1:
                        a = tn.Node(variables[i+1][:,:,0])
                    else:
                        a =tn.Node(variables[i+1])
                    a_list[-1][-1]^ a[0]
                    a_list.append(a)

                #a_list[-1][-1]^a_list[0][0]
                node = a_list[0]
                for i in range(num_node-1):
                    node = tn.contract_between(node,a_list[i+1])
                    
                return node
            def overlap(node1, node2):
                assert len(node1.tensor.shape) == len(node2.tensor.shape)
                num = len(node1.tensor.shape)
                sz = tf.constant([[1,0],[0,-1]])
                sz_t = tn.Node(sz)
                for i in range(num):
                 
                    node1[i]^node2[i]
                return tn.contract_between(node1,node2).tensor
            def overlap_op(node1,node2):
                assert len(node1.tensor.shape) == len(node2.tensor.shape)
                num = len(node1.tensor.shape)
                sz = tf.constant([[1.,0],[0,-1.]])
                sz_t = tn.Node(sz)
                for i in range(num):
                    if i == 1:
                        print(i)
                        node1[i]^ sz_t[0]
                        sz_t[1]^ node2[i]
                    else:
                        node1[i]^node2[i]
                return tn.contract_between(tn.contract_between(node1,sz_t),node2).tensor

            def loss_function(variables, constant):
                num_node, bond_dim, physical_dim = variables.shape[:3]

                nodes1 = build_block(num_node, bond_dim, physical_dim, variables)
                nodes2 = build_block(num_node, bond_dim, physical_dim, variables)
                nodes_norm1 = build_block(num_node, bond_dim, physical_dim, constant)
                nodes_norm2 = build_block(num_node, bond_dim, physical_dim, constant)
                norm = overlap(nodes_norm1, nodes_norm2)
                #print("node is {0}".format(overlap_op(nodes1, nodes2)))
                output = overlap_op(nodes1, nodes2)/norm
                print("norm is {0}".format(norm))
            
                return output
            

            loss = loss_function(self.vars, constant)
            print("loss is {0}".format(loss))
            return loss

    num_node = 10
    bond_dim = 3
    physical_dim = 2



    mean_result = 10**-34
    output_list = []
    for num_node in range(9,12,2):
        print("num_node is {0}".format(num_node))

        constant = tf.constant(tf.random.uniform(shape=(num_node, bond_dim, physical_dim, bond_dim))-0.5)

        do_dx_list = []

        for i in range(2000):
            print("i is {0}".format(i))

            layer1 = TNLayer(num_node, bond_dim, physical_dim)
            #print("layer1 is {0}".format(layer1)) 
            var_constant = copy.copy(layer1.vars)
            with tf.GradientTape(persistent=True) as t:
                t.watch(layer1.vars)
                output = layer1(var_constant)
            do_dx = t.gradient(output, layer1.vars)
            #print(do_dx[0,0,0,0])
            do_dx_list.append(do_dx)
            if i %1000 == 0 and np.abs(np.mean(np.array(do_dx_list)**2)-mean_result)/mean_result<10**-3:

                break
            elif i%1000 == 0:
                print("mean value is {0}".format(mean_result))
                mean_result = np.mean(np.array(do_dx_list)**2) 
        
        output_list.append(np.array(do_dx_list)**2)
        print("number of node is {0}, variance is {1}".format(num_node, output_list[-1]))

        #output_list = np.array(output_list)
        np.save("local_node_{0}".format(num_node),np.array(do_dx_list))
