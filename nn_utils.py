#-*- coding: utf-8 -*-

import sys
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


def create_rnn_cell(num_units, unit_type="gru", num_layers=1, dropout=0.0, mode="train"):
    """Create multi-layer RNN cell."""
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(num_units, unit_type, dropout, mode)
        cell_list.append(single_cell)
    if len(cell_list) == 1:
        return cell_list[0]
    else: 
        return tf.nn.rnn_cell.MultiRNNCell(cell_list)

def _single_cell(num_units, unit_type, dropout, mode):
    """Create an instance of a single RNN cell.""" 
    # Dropout (equal 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == "train" else 0.0
    if unit_type == 'lstm':
        single_cell = tf.nn.rnn_cell.LSTMCell(num_units)
    elif unit_type == "gru":
        single_cell = tf.nn.rnn_cell.GRUCell(num_units)
    if dropout > 0.0:
        single_cell = tf.nn.rnn_cell.DropoutWrapper(
            cell=single_cell, input_keep_prob = (1.0-dropout))
    return single_cell

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

def linear(inputs, output_size, bias=True, concat=True, dtype=None, scope=None):
    """Linear layer

    Args:
        inputs: A Tensor or a list of Tensors with shape [batch, input_size]
        output_size: An integer specify the output size
        bias: a boolean value indicate whether to use bias term
        concat: a boolean value indicate whether to concatenate all inputs
        dtype: an instance of tf.DType, the default value is ``tf.float32''
        scope: the scope of this layer, the default value is ``linear''

    Returns:
        A Tensor with shape [batch, output_size]
    
    Raises:
        RuntimeError: raises ``RuntimeError'' when input sizes do not
                          compatible with each other
    """

    with tf.variable_scope(scope, default_name="Linear", values=[inputs]):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]

        input_size = [item.get_shape()[-1].value for item in inputs]

        if len(inputs) != len(input_size):
            raise RuntimeError("inputs and input_size unmatched!")

        output_shape = tf.concat([tf.shape(inputs[0])[:-1], [output_size]],
                                 axis=0)
        # Flatten to 2D
        inputs = [tf.reshape(inp, [-1, inp.shape[-1].value]) for inp in inputs]

        results = []

        if concat:
            input_size = sum(input_size)
            inputs = tf.concat(inputs, 1)

            shape = [input_size, output_size]
            matrix = tf.get_variable("Matrix", shape, dtype=dtype)
            results.append(tf.matmul(inputs, matrix))
        else:
            for i in range(len(input_size)):
                shape = [input_size[i], output_size]
                name = "matrix_%d" % i
                matrix = tf.get_variable(name, shape, dtype=dtype)
                results.append(tf.matmul(inputs[i], matrix))

        output = tf.add_n(results)

        if bias:
            shape = [output_size]
            bias = tf.get_variable("Bias", shape, dtype=dtype)
            output = tf.nn.bias_add(output, bias)

        output = tf.reshape(output, output_shape)

        return output

def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article
    
    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas

def print_tensors(model_path):
    file_name = "{}/model/model.ckpt".format(model_path) 
    print_tensors_in_checkpoint_file(
        file_name=file_name, tensor_name="", all_tensors=False)

if __name__ == "__main__":
    model_path = sys.argv[1]
    print_tensors(model_path)
