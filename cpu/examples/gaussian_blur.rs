use cuimage::*;
use image::{Rgb, Rgba};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let img = image::open("cpu/examples/lenna.png").unwrap().into_rgba8();

    let context = CuContext::default();
    let src_gpu = CuImage::<Rgba<u8>>::from_host_image(&img)?;
    let mut dst_gpu = CuImage::<Rgb<u8>>::new(img.width(), img.height());

    let blur = GaussianBlur::new(&context, 2);
    blur.run(&src_gpu, &mut dst_gpu)?;

    let out = dst_gpu.to_host_image()?;
    out.save("cpu/examples/lenna_blur.png").unwrap();

    Ok(())
}
