from keras import backend as K
import numpy
from keras.models import Model
import tensorflow as tf
########################
# use to compute saliency scores for each  basepair for the predictions in a certain range
# USE:
# saliency = [compute_gradient(model,sequence_data,predicted_classes,output_index, min=1800, max=1900) for output_index in range(len(predicted_classes))]
# saliency=numpy.vstack(saliency)
# saliency_scores = numpy.sum(saliency,axis=-1)

# sequence_data: sequence data in shape (1, 4000, 4) (4000 is sequence length)
# model: has to be keras model, ie genome_model.get_underlying_network() (make sure before assinging model you
#        have changed the seq_input_legnth to genome_model accordingly, ie. genome_model.change_input_length(sequence_data.shape[1])
# predicted_classes: these are the predicted class labels, ie list of arrays of shape (4000,)

#######################


def get_gradient(model,sequence_data,i, output_index, out_class):
    input_tensors = [model.input, K.learning_phase()]# bassenji uses instead the representation. also change in next line.
    # instead of model.output. bassenji usees activation(representation) to compute the gradient
    model_output=model.output[output_index][i,out_class]
    gradients = model.optimizer.get_gradients(model_output, model.input)
    compute_gradient = K.function(inputs = input_tensors, outputs = gradients)
    return(compute_gradient([sequence_data,0]))

# output_index is the output label to be visualised

def compute_gradient(model, x,output_index = 0,min=None, max=None, input_layer_index=0):
    if len(x.shape)<3:
        x = numpy.expand_dims(x, axis=0)
    else:
        assert x.shape[0]==1, 'Error: can compute saliency scores only for individual sequences.'
    if min is None:
        range_to_compute = tf.range(x.shape[1])
    else:
        range_to_compute=tf.range(min, max+1)
    # computing the saliency map for having predicted class y[output_index][i] at position i
    input_tensors = [model.input, K.learning_phase()]
    model_input = model.layers[input_layer_index].input # the input for convolution layer - can change this to 'representation'?
    # if several outputs choose the one specified and take the first sequence, otherwise just take first sequence -
    # we are computing saliency scores only for one sequence at a time.
    if isinstance(model.output, list):
        model_output = model.output[output_index][0]
    else:
        model_output = model.output[0]
    pred_label =tf.argmax(model_output,axis=-1,output_type=tf.int32)# we are interested in the predicted label
    # pred_label.eval(session=sess)
    cat_idx = tf.stack([ range_to_compute, tf.gather(pred_label, range_to_compute)], axis=1)
    model_output = tf.gather_nd(model_output, cat_idx)

    gradients = model.optimizer.get_gradients(model_output,model_input)
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)
    # Execute the function to compute the gradient

    return compute_gradients([x, 0])[0][0]




# Following adapted from https://github.com/kundajelab/dragonn/tree/master/dragonn

def get_preact_function(model, target_layers):
    # load the model to predict preacts
    preact_model = Model(inputs=model.input,
                         outputs=[l.output for l  in target_layers])
    return preact_model.predict


def in_silico_mutagenesis(model, X,  target_layers,out_index):
    """
    Parameters
    ----------
    model: keras model object
    X: input matrix: (num_samples, sequence_length,num_bases)
    Returns
    ---------
    (num_task, num_samples, sequence_length,num_bases) ISM score array.
    """

    # get the output of the net before the last layer which is the softmax layer.
    preact_function = get_preact_function(model, target_layers)

    # 1. get the wildtype predictions (n,1)

    #wild_type_logits = np.expand_dims(preact_function(X)[:,:, task_index], axis=1)

    # values of wildtype sequence before activation
    # wild_type_preact is of dim (n_samples, seq_len, num_classes)
    wild_type_preact = preact_function(X)
    wild_type_pred = model.predict(X)[out_index]
    seq_len = X.shape[1]
    num_bases = X.shape[-1]
    n_samples = X.shape[0]
    num_classes=wild_type_pred.shape[-1]
    # 2. expand the wt array to dimensions: (n,1,sequence_length,num_bases)
    #X=np.expand_dims(X,1)

    # Initialize mutants array to the same shape



    #output_dim = wild_type_logits.shape + X.shape[3:4]

    # outdim dimensions (num_tasks, num_samples,sequence_length,num_bases)


    # outdim dimensions (num_samples,sequence_length,num_classes,num_bases)
    output_dim= (n_samples,seq_len,num_classes,num_bases)
    # we collect the predictions n mutants expanded for each base letter
    wt_expanded = numpy.empty(output_dim)
    mutants_expanded = numpy.empty(output_dim)
    empty_onehot = numpy.zeros(num_bases,dtype=int)
    # 3. Iterate through all tasks, positions

    mutants=[]
    mutants_pred = []
    mutants_diff=[]
    mutants_pred_diff = []
    for sample_index in range(n_samples):
        print("ISM: sample:" + str(sample_index))
        # fill in wild type logit values into an array of dim (task,sequence_length,num_bases)
        wt_logit_for_task_sample = wild_type_preact[sample_index]
        wt_pred_for_task_sample = wild_type_pred[sample_index]
        #wt_expanded[sample_index] = np.tile(wt_logit_for_task_sample, num_bases)
        # mutagenize each position
        mut = []
        mut_pred = []
        mut_diff = []
        mut_pred_diff = []
        for base_pos in range(seq_len):
            # for each position, iterate through the 4 bases
            base_mut = []
            base_mut_pred = []
            base_mut_diff = []
            base_mut_pred_diff = []
            for base_letter in range(num_bases):
                cur_base = numpy.array(empty_onehot)
                cur_base[base_letter] = 1
                Xtmp = numpy.array(numpy.expand_dims(X[sample_index], axis=0))
                # mutate base at base_pos:
                Xtmp[0][base_pos] = cur_base
                # get the pre-activation output for Xtmp
                Xtmp_logit = numpy.squeeze(preact_function(Xtmp), axis=0)
                Xtmp_pred = model.predict(Xtmp)[out_index]
                # subtract wt prediction from mutants prediction
                ism_val = Xtmp_logit
                ism_val_diff = Xtmp_logit- wt_logit_for_task_sample
                ism_pred = numpy.squeeze(numpy.argmax(Xtmp_pred,-1))
                ism_pred_diff = numpy.squeeze(numpy.argmax(Xtmp_pred, -1) - numpy.argmax(wt_pred_for_task_sample, -1))
                if base_pos==0 and base_letter==0 and sample_index==0:
                    ism_val_0=ism_val
                base_mut.append(ism_val)
                base_mut_pred.append(ism_pred) # contains what are different predictions if base at pos base_pos is mutated to base_letter
                base_mut_diff.append(ism_val_diff)
                base_mut_pred_diff.append(ism_pred_diff)

            mut.append(numpy.stack(base_mut,2)) # list of (seqlen,4) containing different predictions if base is mutated to other bases going from base pos 0 to base pos seqlen
            mut_pred.append(numpy.stack(base_mut_pred, 1))# mutation of 0ths base to base (1,0,0,0) is in mut[0][:,:,0] e.g. mut[0][:,:,0]=ism_val_0 for ism_val_0 is value for base_pos = base_letter=0

            mut_diff.append(numpy.stack(base_mut_diff,
                               2))  # list of (seqlen,4) containing different predictions if base is mutated to other bases going from base pos 0 to base pos seqlen
            mut_pred_diff.append(numpy.stack(base_mut_pred_diff, 1))
        mutants.append(numpy.stack(mut,0)) #mutants[0][0,:,:,0]==ism_val_0
        mutants_pred.append(numpy.stack(mut_pred, 0))
        mutants_diff.append(numpy.stack(mut_diff,0)) #mutants[0][0,:,:,0]==ism_val_0
        mutants_pred_diff.append(numpy.stack(mut_pred_diff, 0))
        #entry of mutants is of dim (seq_len,seq_len,num_classes, n_bases): per sample for each base position the seqlenbp prediction for respective base

    # ism_vals is of dim (n-samples,seq_len,seq_len,num_classes, n_bases)
    ism_vals = numpy.stack(mutants,0)
    ism_preds = numpy.stack(mutants_pred, 0)
    ism_vals_diff = numpy.stack(mutants_diff,0)
    ism_preds_diff = numpy.stack(mutants_pred_diff, 0)
    return ism_preds_diff, ism_preds, ism_vals, ism_vals_diff



