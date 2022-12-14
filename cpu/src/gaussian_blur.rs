use super::buffer::CuImage;
use super::context::CuContext;
use cust::memory::DeviceCopy;
use cust::prelude::*;
use image::{Pixel, Rgb};
use std::error::Error;
use std::marker::PhantomData;

pub struct GaussianBlur<'a, P: Pixel>
where
    <P as Pixel>::Subpixel: DeviceCopy,
{
    context: &'a CuContext,
    weights: DeviceBuffer<f32>,
    sigma: i32,
    _phantom: PhantomData<P>,
}

impl<'a, P: Pixel> GaussianBlur<'a, P>
where
    <P as Pixel>::Subpixel: DeviceCopy,
{
    pub fn new(context: &'a CuContext, sigma: i32) -> Self {
        let weight_size = (sigma * 2 + 1) * (sigma * 2 + 1);
        let mut weights: Vec<f32> = vec![0.0; weight_size as usize];
        for i in -sigma..(sigma + 1) {
            for j in -sigma..(sigma + 1) {
                let weight = (-((i * i + j * j) as f32) / (2.0 * ((sigma * sigma) as f32))).exp();
                weights[(i + sigma + (j + sigma) * (sigma * 2 + 1)) as usize] = weight;
            }
        }
        let sum_weight = weights.iter().sum::<f32>();
        for i in 0..weight_size {
            weights[i as usize] /= sum_weight;
        }
        GaussianBlur {
            context,
            weights: weights.as_slice().as_dbuf().unwrap(),
            sigma,
            _phantom: PhantomData,
        }
    }
    pub fn run(
        &self,
        src_img: &CuImage<P>,
        dst_img: &mut CuImage<Rgb<u8>>,
    ) -> Result<(), Box<dyn Error>> {
        let kernel = self.context.module.get_function("gaussian_blur").unwrap();
        let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (src_img.width * src_img.height + block_size - 1) / block_size;
        let stream = &self.context.stream;
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    src_img.data.as_device_ptr(),
                    src_img.width * src_img.height,
                    self.weights.as_device_ptr(),
                    self.weights.len(),
                    dst_img.data.as_device_ptr(),
                    src_img.width,
                    src_img.height,
                    src_img.num_of_channels,
                    self.sigma,
                )
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }
}
