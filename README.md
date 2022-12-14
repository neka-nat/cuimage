# cuimage


Rust implementation of image processing library with CUDA.

## Core feature

* Use [Rust-CUDA](https://github.com/Rust-GPU/Rust-CUDA)
* Implemented algorithms
  * Convert gray scale
  * Gaussian blur


## Build

```sh
docker build -t rust-cuda .
```

```sh
docker run -it --gpus all -v $PWD:/root/rust-cuda --entrypoint /bin/bash rust-cuda
cd /root/rust-cuda
cargo run --release --example gray
```

### Original image
![original](cpu/examples/lenna.png)

### Output images
![gray](asset/lenna_gray.png)
![blur](asset/lenna_blur.png)
