use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../gpu")
        .copy_to("../resources/cuimage_gpu.ptx")
        .build()
        .unwrap();
}
