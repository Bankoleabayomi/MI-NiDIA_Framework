Here are important things to note for effective appication of MI-NiDIA Software.
1. The software has only been validated with varying floc length sizes under different flocculation conditions (data from Non-Intrusive Dynamic Image Analysis),
   so data input should be floc data containing either a group or multiple groups of floc count per Tf.
2. Your input does not need to be scaled as the software will upscale your dataset, as such, a column for your original timestamp (Tf) should be included and labelled as Tf in the datasheet.
3. Users are encourage to downgrade their Tensorflow and Keras version (to 2.12.0 to 2.15.0) in a situation where the commands used in this code are no longer compatible with the latest version of Tensorflow.
4. In an instance where the software is to be run on a system with lesser GPU power, the n_job is advised to be set as 1 (i.e. n_job = 1) in the minidiaModel.py file.

