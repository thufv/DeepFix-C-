import tensorflow as tf

def embedding_attention_bidirectional_seq2seq(encoder_inputs, decoder_inputs, input_cell1, input_cell2, output_cell, num_encoder_symbols,
    num_decoder_symbols, embedding_size, num_heads=1, output_projection=None, feed_previous=False, dtype=None, scope=None,
    initial_state_attention=False):

    with tf.variable_scope(scope or "embedding_attention_bidirectional_seq2seq") as scope:
        # Encoder.
        encoder_cell1 = tf.nn.rnn_cell.EmbeddingWrapper(input_cell1, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
        encoder_cell2 = tf.nn.rnn_cell.EmbeddingWrapper(input_cell2, embedding_classes=num_encoder_symbols, embedding_size=embedding_size)
        encoder_outputs, encoder_state1, encoder_state2 = tf.nn.bidirectional_rnn(encoder_cell1, encoder_cell2, encoder_inputs, dtype=tf.float32)
        encoder_state = tf.concat(1, [encoder_state1, encoder_state2])

        # First calculate a concatenation of encoder outputs to put attention on.
        top_states = [tf.reshape(e, [-1, 1, input_cell1.output_size + input_cell2.output_size]) for e in encoder_outputs]
        attention_states = tf.concat(1, top_states)

        # Decoder.
        output_size = None
        if output_projection is None:
            output_cell = tf.nn.rnn_cell.OutputProjectionWrapper(output_cell, num_decoder_symbols)
            output_size = num_decoder_symbols

        assert isinstance(feed_previous, bool)
        return tf.nn.seq2seq.embedding_attention_decoder(decoder_inputs, encoder_state, attention_states, output_cell,
            num_decoder_symbols, embedding_size, num_heads=num_heads, output_size=output_size, output_projection=output_projection,
            feed_previous=feed_previous, initial_state_attention=initial_state_attention)
