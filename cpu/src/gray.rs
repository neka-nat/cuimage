use super::buffer::CuImage;
use super::context::CuContext;
use cust::memory::DeviceCopy;
use cust::prelude::*;
use image::{Luma, Pixel};
use std::error::Error;
use std::marker::PhantomData;

pub struct Gray<'a, P: Pixel>
where
    <P as Pixel>::Subpixel: DeviceCopy,
{
    context: &'a CuContext,
    _phantom: PhantomData<P>,
}

impl<'a, P: Pixel> Gray<'a, P>
where
    <P as Pixel>::Subpixel: DeviceCopy,
{
    pub fn new(context: &'a CuContext) -> Self {
        Gray {
            context,
            _phantom: PhantomData,
        }
    }
    pub fn run(
        &self,
        src_img: &CuImage<P>,
        dst_img: &mut CuImage<Luma<u8>>,
    ) -> Result<(), Box<dyn Error>> {
        let kernel = self.context.module.get_function("gray").unwrap();
        let (_, block_size) = kernel.suggested_launch_configuration(0, 0.into())?;
        let grid_size = (src_img.width * src_img.height + block_size - 1) / block_size;
        let stream = &self.context.stream;
        unsafe {
            launch!(
                kernel<<<grid_size, block_size, 0, stream>>>(
                    src_img.data.as_device_ptr(),
                    src_img.width * src_img.height,
                    dst_img.data.as_device_ptr(),
                    src_img.num_of_channels,
                )
            )?;
        }
        stream.synchronize()?;
        Ok(())
    }
}
