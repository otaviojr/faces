use ffmpeg_next::format::{input, Pixel};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context, flag::Flags};
use ffmpeg_next::util::frame::video::Video;

use std::fs::File;
use std::io::prelude::*;

pub struct Camera {
    pub url: String,
    pub scale_width: f32,
    pub scale_height: f32,
}

impl Camera {
    pub fn new(url: &str, scale_width: f32, scale_height: f32) -> Camera {

        return Camera {
          url: String::from(url),
          scale_width,
          scale_height,
        };
    }

    pub fn connect<F>(&self, callback: F) -> Result<(), ffmpeg_next::Error>
        where F: Fn(&mut Video, u32, u32, u32, u32) -> i32, {
        let mut ictx = input(&self.url)?;

        let input = ictx
        .streams()
        .best(Type::Video)
        .ok_or(ffmpeg_next::Error::StreamNotFound)?;

        let video_stream_index = input.index();

        let context_decoder = ffmpeg_next::codec::context::Context::from_parameters(input.parameters())?;
        let mut decoder = context_decoder.decoder().video()?;

        let width = (self.scale_width * decoder.width() as f32).round() as u32;
        let height = (self.scale_height * decoder.height() as f32).round() as u32;

        let mut scaler = Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            Pixel::RGB24,
            width,
            height,
            Flags::BILINEAR,
        )?;

        let mut idx = 0;

        let mut receive_and_process_decoded_frames = |decoder: &mut ffmpeg_next::decoder::Video| -> Result<(), ffmpeg_next::Error> {
            let mut decoded = Video::empty();
            while decoder.receive_frame(&mut decoded).is_ok() {
                idx += 1;
                let mut rgb_frame = Video::empty();
                scaler.run(&decoded, &mut rgb_frame)?;
                if idx % 10 == 0{
                    callback(&mut rgb_frame, width,height, decoder.width(), decoder.height());
                }
            }
            Ok(())
        };

        for (stream, packet) in ictx.packets() {
            if stream.index() == video_stream_index {
                decoder.send_packet(&packet)?;
                receive_and_process_decoded_frames(&mut decoder)?;
            }
        }

        decoder.send_eof()?;
        receive_and_process_decoded_frames(&mut decoder)?;
        Ok(())
    }

    pub fn save_file(frame: &[u8], width: u32, height: u32, index: usize) -> std::result::Result<(), std::io::Error> {
        let mut file = File::create(format!("frame{}.ppm", index))?;
        file.write_all(format!("P6\n{} {}\n255\n", width, height).as_bytes())?;
        file.write_all(frame)?;
        Ok(())
    }
}

