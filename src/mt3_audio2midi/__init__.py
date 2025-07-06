import functools
import numpy as np
import tensorflow as tf
import librosa
import note_seq
from importlib import resources
import gin
import jax
import seqio
import t5
import mt3_audio2midi.mt3.note_sequences
import mt3_audio2midi.mt3.vocabularies
import mt3_audio2midi.mt3.spectrograms
import mt3_audio2midi.mt3.models
import mt3_audio2midi.mt3.network
import mt3_audio2midi.mt3.preprocessors
import mt3_audio2midi.mt3.metrics_utils
import mt3_audio2midi.t5x.partitioning
import mt3_audio2midi.t5x.utils
import mt3_audio2midi.t5x.adafactor

class MT3():
	def __init__(self, model_path, model_type='mt3'):
		if model_type == 'ismir2021':
			num_velocity_bins = 127
			self.encoding_spec = mt3_audio2midi.mt3.note_sequences.NoteEncodingSpec
			self.inputs_length = 512
		elif model_type == 'mt3':
			num_velocity_bins = 1
			self.encoding_spec = mt3_audio2midi.mt3.note_sequences.NoteEncodingWithTiesSpec
			self.inputs_length = 256
		else:
			raise ValueError('unknown model_type: %s' % model_type)
		self.batch_size = 8
		self.outputs_length = 1024
		self.sequence_length = {'inputs': self.inputs_length,'targets': self.outputs_length}
		self.partitioner = mt3_audio2midi.t5x.partitioning.PjitPartitioner(model_parallel_submesh=None, num_partitions=1)
		self.spectrogram_config = mt3_audio2midi.mt3.spectrograms.SpectrogramConfig()
		self.codec = mt3_audio2midi.mt3.vocabularies.build_codec(vocab_config=mt3_audio2midi.mt3.vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins))
		self.vocabulary = mt3_audio2midi.mt3.vocabularies.vocabulary_from_codec(self.codec)
		self.output_features = {'inputs': seqio.ContinuousFeature(dtype=tf.float32, rank=2),'targets': seqio.Feature(vocabulary=self.vocabulary),}
		package_dir = resources.files(__package__)
		with gin.unlock_config():
			gin.parse_config_files_and_bindings([package_dir.joinpath("gin","model.gin"),package_dir.joinpath("gin",f"{model_type}.gin")], ['from __gin__ import dynamic_registration','from mt3_audio2midi.mt3 import vocabularies','VOCAB_CONFIG=@vocabularies.VocabularyConfig()','vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS'], finalize_config=False)
		self.model = mt3_audio2midi.mt3.models.ContinuousInputsEncoderDecoderModel(module=mt3_audio2midi.mt3.network.Transformer(config=gin.get_configurable(mt3_audio2midi.mt3.network.T5Config)()),input_vocabulary=self.output_features['inputs'].vocabulary,output_vocabulary=self.output_features['targets'].vocabulary,optimizer_def=mt3_audio2midi.t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),input_depth=mt3_audio2midi.mt3.spectrograms.input_depth(self.spectrogram_config))
		train_state_initializer = mt3_audio2midi.t5x.utils.TrainStateInitializer(optimizer_def=self.model.optimizer_def,init_fn=self.model.get_initial_variables,input_shapes={'encoder_input_tokens': (self.batch_size, self.inputs_length),'decoder_input_tokens': (self.batch_size, self.outputs_length)},partitioner=self.partitioner)
		self._predict_fn = self._get_predict_fn(train_state_initializer.train_state_axes)
		self._train_state = train_state_initializer.from_checkpoint_or_scratch([mt3_audio2midi.t5x.utils.RestoreCheckpointConfig(path=model_path, mode='specific', dtype='float32')], init_rng=jax.random.PRNGKey(0))

	def _get_predict_fn(self, train_state_axes):
		def partial_predict_fn(params, batch, decode_rng):
			return self.model.predict_batch_with_aux(params, batch, decoder_params={'decode_rng': None})
		return self.partitioner.partition(partial_predict_fn,in_axis_resources=(train_state_axes.params,mt3_audio2midi.t5x.partitioning.PartitionSpec('data',), None),out_axis_resources=mt3_audio2midi.t5x.partitioning.PartitionSpec('data',))

	def preprocess(self, ds):
		for pp in [functools.partial(t5.data.preprocessors.split_tokens_to_inputs_length,sequence_length=self.sequence_length,output_features=self.output_features,feature_key='inputs',additional_feature_keys=['input_times']),mt3_audio2midi.mt3.preprocessors.add_dummy_targets,functools.partial(mt3_audio2midi.mt3.preprocessors.compute_spectrograms,spectrogram_config=self.spectrogram_config)]:
			ds = pp(ds)
		return ds

	def postprocess(self, tokens, example):
		if mt3_audio2midi.mt3.vocabularies.DECODED_EOS_ID in np.array(tokens, np.int32):
			tokens = tokens[:np.argmax(tokens == mt3_audio2midi.mt3.vocabularies.DECODED_EOS_ID)]
		start_time = example['input_times'][0]
		return {'est_tokens': tokens,'start_time': start_time - start_time % (1 / self.codec.steps_per_second),'raw_inputs': []}

	def predict(self, audio_path, seed=0,output_file="output.mid"):
		audio = librosa.load(audio_path,sr=16000)[0]
		frame_size = self.spectrogram_config.hop_width
		padding = [0, frame_size - len(audio) % frame_size]
		audio = np.pad(audio, padding, mode='constant')
		frames = mt3_audio2midi.mt3.spectrograms.split_audio(audio, self.spectrogram_config)
		num_frames = len(audio) // frame_size
		frame_times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
		ds =  tf.data.Dataset.from_tensors({'inputs': frames,'input_times': frame_times,})
		ds = self.preprocess(ds)
		model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length)
		model_ds = model_ds.batch(self.batch_size)
		inferences = (tokens for batch in model_ds.as_numpy_iterator() for tokens in self.vocabulary.decode_tf(self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(seed))[0]).numpy())
		predictions = []
		for example, tokens in zip(ds.as_numpy_iterator(), inferences):
			predictions.append(self.postprocess(tokens, example))
		result = mt3_audio2midi.mt3.metrics_utils.event_predictions_to_ns(predictions, codec=self.codec, encoding_spec=self.encoding_spec)
		note_seq.sequence_proto_to_midi_file(result['est_ns'], output_file)
		return output_file
