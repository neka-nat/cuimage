use cuimage::*;
use image::{Luma, Rgba};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let img = image::open("cpu/examples/lenna.png").unwrap().into_rgba8();

    let context = CuContext::default();
    let src_gpu = CuImage::<Rgba<u8>>::from_host_image(&img)?;
    let mut dst_gpu = CuImage::<Luma<u8>>::new(img.width(), img.height());

    let gray = Gray::new(&context);
    gray.run(&src_gpu, &mut dst_gpu)?;

    let out = dst_gpu.to_host_image()?;
    out.save("cpu/examples/lenna_gray.png").unwrap();

    Ok(())
}
