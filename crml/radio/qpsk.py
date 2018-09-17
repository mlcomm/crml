import math
import numpy as np
import random
from gnuradio import gr, digital
from gnuradio import analog
from gnuradio import blocks
from gnuradio.filter import firdes
from gnuradio import filter

def qpsk_awgn_generator(dataset_size = 10000, 
                        vec_length = 128, 
                        EbN0 = -6):
    
    N_BITS = dataset_size*vec_length
        
    tb = gr.top_block()
    
    const = digital.qpsk_constellation()
    rrc_taps = firdes.root_raised_cosine(1, 4, 1, 0.35, 45)
    
    src = blocks.vector_source_b(map(int, np.random.randint(0, const.arity(), N_BITS/const.bits_per_symbol())), False)
    
    mod = digital.chunks_to_symbols_bc((const.points()), 1)
    match_filter = filter.interp_fir_filter_ccc(4, (rrc_taps))
    amplitude = blocks.multiply_const_vcc((4, ))
    noise_amplitude = 1.0 / math.sqrt(10**(float(EbN0)/10))
    noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amplitude, 0)
    add = blocks.add_vcc(1)
    sink = blocks.vector_sink_c()
    noise_sink = blocks.vector_sink_c()
    
    tb.connect(src, mod, match_filter, amplitude, (add, 0), sink)
    tb.connect(noise, (add, 1))
    tb.connect(noise, noise_sink)
    tb.run()
    
    sample_output = np.array(sink.data(), dtype=np.complex64)
    noise_output = np.array(noise_sink.data(), dtype=np.complex64)
    
    sampler_indx = random.randint(50, 500)
    vec_indx = 0
    
    sample_data = np.zeros([dataset_size, 2, vec_length], dtype=np.float32)
    noise_data = np.zeros([dataset_size, 2, vec_length], dtype=np.float32)
    sample_labels = np.zeros([dataset_size, 2], dtype=np.int32)
    noise_labels = np.zeros([dataset_size, 2], dtype=np.int32)
    
    while sampler_indx + vec_length < len(sample_output) and vec_indx < dataset_size:
        sampled_vector = sample_output[sampler_indx:sampler_indx+vec_length]
        sample_data[vec_indx, 0, :] = np.real(sampled_vector)
        sample_data[vec_indx, 1, :] = np.imag(sampled_vector)
        
        noise_vector = noise_output[sampler_indx:sampler_indx+vec_length]
        noise_data[vec_indx, 0, :] = np.real(noise_vector)
        noise_data[vec_indx, 1, :] = np.imag(noise_vector)
        
        sample_labels[vec_indx, :] = [1 , 0]
        noise_labels[vec_indx, :] = [0 , 1]
       
        sampler_indx += vec_length
        vec_indx += 1

    data = []
    data.append(sample_data)
    data.append(noise_data)
    data = np.vstack(data)
    
    labels = []
    labels.append(sample_labels)
    labels.append(noise_labels)
    labels = np.vstack(labels)
    
    return data, labels, vec_indx
