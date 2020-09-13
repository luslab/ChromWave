import copy



from ..functions import utils

__all__=[
    'get_motif',
    'return_sliced_sequence'
]


def return_sliced_sequence(sequence,start_list,end_list):

    if len(start_list)!= len(end_list):
        print('error')
    else:
        to_return=[sequence[:,start_list[i]:end_list[i]] for i in range(len(start_list))]
        return to_return

# Returns a list of sequences of X[instance] that gives max in motif detector.
# If in motif detector all values are 0, then it returns array with all=0.25


    # if you don't use copy.copy the list will not be properly cloned (some python internal shit) and
def get_motif(X_i, start_x_i,end_x_i, max_list_i,  kernel_length, conv_n_filters,thresh=0):
    # output of conv layer of instance
    # getting the start index of the sequence of interest

    # we need to check that the max is actually >0
    # if you don't use copy.copy the list will not be properly cloned (some python internal shit) and
    pads = (kernel_length - 1)
    motif_list = [
        utils.padding_dna(copy.copy(X_i), pads, pads)[:, start_x_i[i]:end_x_i[i]] for
        i in range(start_x_i.shape[-1])]

    # return_sliced_sequence(copy.copy(X[instance]),start_index,end_index)
    # loop over the feature maps
    for motif_k in range(conv_n_filters):
        # print X[instance].sum(axis=0)
        if max_list_i[motif_k] <= thresh[motif_k]:
            motif_list[motif_k].fill(0.0)
            # print X[instance].sum(axis=0)
            # replace 0.25 with 0
        motif_list[motif_k][motif_list[motif_k] < 1] = 0

    # print X[instance].sum(axis=0)
    return motif_list
