## Discussion

  * **Moderation**: The weight from moderators tell us how much we should trust
    the feature (information) from body image or part (head/hands) crops, the
    animation figure shows the visualization of the weights: the lighter color
    means it trust body more.  
  * **Face from SMPL-X model**: In PIXIE, we use a common shape variable for the body and the face.
    The shape space of SMPL-X is learned from 
    3800 real full body scans and captures the correlation between body and
    face shape. In the samples script we support the option of predicting 
    the shape of the full body from a face-only image. If you are only interested in
    getting accurate face shape,
    we suggest to try other face-specifc work, such as [DECA](https://github.com/YadiraF/DECA).  
 * **Model conversion**: If you want to convert the predicted SMPL-X body to a different body model, 
    e.g. SMPL, please take a look [here](https://github.com/vchoutas/smplx/blob/master/transfer_model/README.md)

### Limitations:

  * **Cropping matters:** Even though we already did a lot of data
    augmentations durining training, the results will still vary a bit due to
    cropping differences for the input images. For
    simplicity, we chose Faster-RCNN as a person detector, 
    while in the paper we used OpenPose keypoints to compute a bounding box.  
  * **Perspective projection**: We use a scaled-orthographic/weak-perspective camera model in PIXIE, which
      does not work well for images with strong perpective distortion.  
  * **Misalignment issue**: For regression works that output model
    (SMPL/SMPL-X) parameters, misalignment with the person in the image is always an issue.
    This can be improved by using PIXIE results as initialization for an optimization method
    like [SMPLify-X](https://github.com/vchoutas/smplify-x)
    to refine the pose. Note that, the moderator weight could also be utilized
    as a confidence measure during optimization.  
  * **Speed**: The main bottleneck of PIXIE is the need for three separate encoders
    for the body, head and hand images. Changing the current
    backbone, i.e. Resnet50 or HRNet, to a lighter one, like MobileNet, should
    accelerate inference at the cost of performance. We will attempt to provide different
    options when we release the **training** code. 
  * **Texture**: Similar to [DECA](https://github.com/YadiraF/DECA), we rely
    upon the Basel face model for our albedo space. Its lack of ethnic
    diversity in the albedo causes the model to often compensate for skin tone with lighting.
