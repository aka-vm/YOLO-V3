# YOLOv3

## Introduction

## Implimentation

In this Implimentation, things are made configureable to a point that calling them **Darknet-53** or **YOLOv3** would be inaccurate.
Things only vary when it comes to the exact shape of input and output tensors. The main idea of using the input image of shape `(Iʷ, Iᴴ, 3)`, `3 Prediction Scales` of shape `(Sʷ₁, Sʰ₁`, `(Sʷ₂, Sʰ₂`, and `(Sʷ₃, Sʰ₃`, and `3 Anchor Boxes` per prediction scales remains the same.

```
Input Image Shape - (m, Iʷ, Iᴴ, 3)
Output - Tensor(m, Sʷᵢ, Sʰᵢ, 3, 5+Cⁿ) ∀ i ∈ {1,2,3}

(Iʷ, Iᴴ) == 32*S₁                                      Original- (416, 416)
(S₁,S₂,S₃) == S₁, 2*S₁, 4*S₁                           Original- (13, 26, 52)
```


### Network
<!--
* Reusable Blocks/Layers
  * Conv
  * Residual
  * ConvPass
  * Output
* Model
  * I/O
  * Build and call
  * Verbose
    * Summary
    * Sketch
-->


### Data Loader
<!--
* Loading Data -> (Image, File)
* Augmentation -> TODO
* Format Bounding Boxes and adding targets
  * Input -> str(C x y w h)
        # 11 0.341 0.609 0.416 0.262
  * Output
    * Shape -> (m, Sʷᵢ, Sʰᵢ, 3, 6) # (m, 13, 13, 3, 6)
    * Managing Anchor Box -> (3,3,2)
      # No need to manage them here.
    * data -> (Obj,c_x, c_y, c_w, c_h, C)
              # (1,0.433,0.917,5.408,3.406,11) at (m, 4, 7, A_b)
              Obj - int[-1,1]; 0 if no obj else 1 if best anchor else -1;
              b_x - (0,1); (S*x)-int(S*x) at Sʷᵢ == int(Sʷᵢ*x)
              b_y - (0,1); (S*y)-int(S*y) at Sʰᵢ == int(Sʰᵢ*y)
              b_w - (0,Sʷᵢ); w * Sʷᵢ
              b_h - (0,Sʰᵢ); h * Sʰᵢ
              C   - int[0,Cⁿ); C

              A_b -> the anchor box with best IoU;
              IoU ((w,h), A)->  I/U
                I = min(Aw, w)*min(Ah, h);
                U = wh * prod(A)

* Output Data in Batches of m
 -->

### Loss
<!--
* No Obj Loss
* Obj Loss
  * Box Coordinate Loss
  * Class Loss
* combile all with different λ for each loss
 -->


### Training

<!-- ## Model -->

