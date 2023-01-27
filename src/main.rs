use std::{sync::{Mutex,Arc}, fs::{self, DirEntry}, path::PathBuf, time::Instant, io::Cursor};
use clap::{App, Arg, Values};
use image::{GenericImage, imageops, ImageBuffer, Rgb, io::{Reader}, ImageFormat, EncodableLayout, DynamicImage};
use rand::thread_rng;
use rand::seq::SliceRandom;

use ffmpeg_next::format::{input, Pixel};
use ffmpeg_next::media::Type;
use ffmpeg_next::software::scaling::{context::Context, flag::Flags};
use ffmpeg_next::util::frame::video::Video;

use magick_rust::{PixelWand,DrawingWand,MagickWand, magick_wand_genesis};

pub mod camera;

use std::env;

use neuron::{Neuron, math::Tensor, layers::{ConvLayer, ConvLayerConfig, FlattenLayer, PoolingLayer, PoolingLayerConfig, DenseLayer, DenseLayerConfig}, activations::{ReLU, Sigmoid, Tanh, SoftMax}, cost::Functions, pipeline::SequentialPieline, Loader, util::LogLevel};

static urls: [&str;4] = [
                "rtsp://{USER}:{PASSWORD}@192.168.0.103/cam/realmonitor?channel=1&subtype=1",
                "rtsp://{USER}:{PASSWORD}@192.168.0.104/cam/realmonitor?channel=1&subtype=1",
                "rtsp://{USER}:{PASSWORD}@192.168.0.105/cam/realmonitor?channel=1&subtype=1",
                "rtsp://{USER}:{PASSWORD}@192.168.0.106/cam/realmonitor?channel=1&subtype=1",
                ];

fn main() {

    let mut neuron = Neuron::new();
    let mut pipeline = SequentialPieline::new();

    let img_size: usize = 32;

    let matches = App::new("Faces")
    .version("1.0")
    .author("Otávio Ribeiro")
    .about("Cliente de teste para o neuron-rs")
    .arg(Arg::with_name("train")
        .short("t")
        .long("train")
        .value_name("FILE")
        .help("Train the network before inference")
        .takes_value(false))
    .arg(Arg::with_name("opencl")
        .long("opencl")
        .help("Enable OpenCL")
        .takes_value(false))
    .arg(Arg::with_name("weigths")
        .short("w")
        .long("weights")
        .value_name("FILE")
        .help("Weights file to load")
        .takes_value(true))
    .arg(Arg::with_name("log-level")
        .long("log-level")
        .value_name("LEVEL")
        .help("Set the log level")
        .possible_values(&["debug", "profiling", "info", "warn", "error"])
        .takes_value(true))
    .arg(Arg::with_name("INPUT")
        .help("Image file to make the inference")
        .multiple(false)
        .index(1))
    .get_matches();

    let train = matches.is_present("train");
    let weigths = matches.value_of("weigths").unwrap_or("./weights/faces.weights");
    let input_video = matches.value_of("INPUT").unwrap_or("0");
    let opencl = matches.is_present("opencl");

    let log_level_param = matches.value_of("log-level").unwrap_or("info");

    let log_level = match log_level_param {
        "debug" => LogLevel::Debug,
        "profiling" => LogLevel::Profiling,
        "info" => LogLevel::Info,
        "warn" => LogLevel::Warn,
        "error" => LogLevel::Error,
        _ => LogLevel::Info,
    };

    Neuron::set_log(None, log_level);

    if opencl {
        Neuron::enable_opencl();
    }

    pipeline.add_layer(Mutex::new(Box::new(ConvLayer::new("conv1".to_owned(), 3, 14, (5,5), ConvLayerConfig { activation: Arc::new(ReLU::new()), learn_rate: 10e-6, padding: 0, stride: 1 }))));
    pipeline.add_layer(Mutex::new(Box::new(PoolingLayer::new((2,2), PoolingLayerConfig { stride: 2}))));
    pipeline.add_layer(Mutex::new(Box::new(FlattenLayer::new())));
    pipeline.add_layer(Mutex::new(Box::new(DenseLayer::new("lin1".to_owned(),2744, 2744, DenseLayerConfig{ activation: Arc::new(ReLU::new()), learn_rate: 10e-6}))));
    pipeline.add_layer(Mutex::new(Box::new(DenseLayer::new("lin2".to_owned(),2744, 2, DenseLayerConfig{ activation: Arc::new(SoftMax::new()), learn_rate: 10e-6}))));

    neuron.add_pipeline(Mutex::new(Box::new(pipeline)));

    if let Err(error) = neuron.load(weigths) {
        println!("Error load weigths: ${}", error);
    }

    if train {

        let dir = "./data/";
        let final_dir = "./processed";
        
        let mut vec:Vec<Result<DirEntry,_>> = fs::read_dir(dir).unwrap().collect();
        let mut rng = thread_rng();
        vec.shuffle(&mut rng);

        for _ in 0 .. 1 {
            for (idx,entry) in vec.iter().enumerate() {
                let entry = entry.as_ref().unwrap();
                let path = entry.path();
                let filename = path.file_name().unwrap();
                let filename_string = filename.to_os_string();
                //if filename.to_string_lossy().starts_with("0_") {
                //    continue;
                //}
                //println!("{}", filename.to_string_lossy());
                let p = path.as_path();
                let img_ret = image::open(p);

                if let Err(err) = img_ret {
                    continue;
                }

                let mut img = img_ret.unwrap();
                img = img.resize_to_fill(img_size as u32, img_size as u32, imageops::FilterType::Nearest);

                let image = img.as_rgb8();
                if image.is_none() {
                    let _ = fs::remove_file(p);
                    continue;
                }
        
                let image = image.unwrap();
                    
                let data = img.as_bytes();
                let width = img.width();
        
                let mut r = Vec::new();
                let mut g = Vec::new();
                let mut b = Vec::new();
        
                for row in data.chunks(width as usize * 3) {
                    for pixel in row.chunks_exact(3) {
                        r.push(pixel[0] as f32 / 255.0 * 10e-1);
                        g.push(pixel[1] as f32 / 255.0 * 10e-1);
                        b.push(pixel[2] as f32 / 255.0 * 10e-1);
                    }
                }

                let mut input = Vec::new();
                input.push(Box::new(Tensor::from_data(img_size,img_size,r)));
                input.push(Box::new(Tensor::from_data(img_size,img_size,g)));
                input.push(Box::new(Tensor::from_data(img_size,img_size,b)));
                    
                let label_value = filename_string.to_str().unwrap().chars().nth(0).unwrap();
                let label_digit = label_value.to_digit(0x10).unwrap() as f32;
        
                println!("Starting Forward Propagation");
                let output = neuron.forward(input.clone());
                if let Some(ref y) = output {
                    println!("Output Size={}x{}", y[0].rows(), y[0].cols());
                    println!("Output={:?} - {} - {} - {}", y[0], idx, label_digit, filename_string.to_string_lossy());
                    println!("Cost: {:?} - {} - {}", Functions::binary_cross_entropy_loss(&y[0], &Tensor::from_data(1,1,vec![label_digit])), idx, filename_string.to_string_lossy());
                    println!("Starting Backward Propagation");
                    //let loss = Functions::binary_cross_entropy_loss_derivative(&y[0], &Tensor::from_data(2,1,vec![if label_digit == 1.0 {1.0} else {y[0].get(0,0)}, if label_digit == 0.0 {1.0} else {y[0].get(1,0)}]));
                    let loss = y[0].sub(&Tensor::from_data(2,1,vec![if label_digit == 1.0 {1.0} else {0.0}, if label_digit == 0.0 {1.0} else {0.0}])).unwrap();
                    //let loss = Functions::softmax_loss(&y[0],&Tensor::from_data(2,1,vec![if label_digit == 1.0 {1.0} else {0.0}, if label_digit == 0.0 {1.0} else {0.0}]));
                    println!("Loss: {:?}", loss);
                    //let loss = Functions::binary_cross_entropy_loss_derivative(&y[0], &Tensor::from_data(1,1,vec![label_digit]));
                    //let loss = y[0].sub(&Tensor::from_data(1,1,vec![label_digit]));
                    //if label_digit == 1.0 {
                        neuron.backward(vec![Box::new(loss)]);
                    //}
                }

                if (idx > 0 && idx % 100 == 0) || vec.len() <= 10{
                    println!("Saving Weigths");
                    if let Err(error) = neuron.save(weigths) {
                        println!("Error saving weigths: ${}", error);
                    }
                }

                //move file to final_dir
                let mut final_path = PathBuf::from(final_dir);
                final_path.push(filename_string);
                let _ = fs::rename(p, final_path);
                
            }
        }
    }

    let user = env::var("CAMERA_USER").expect("CAMERA_USER not set");
    let password = env::var("CAMERA_PASSWORD").expect("CAMERA_PASSWORD not set");

    let initial_url = urls[input_video.parse::<usize>().unwrap()];
    let url = initial_url.replace("{USER}", &user).replace("{PASSWORD}", &password);

    let cam = camera::Camera::new(&url, 0.6,0.6);

    let mut idx = Mutex::new(0);

    if let Err(err) = cam.connect(  |rgb_frame: &mut Video, dst_width, dst_height, orig_width, orig_height| {
        let start = Instant::now();

        let width = rgb_frame.width();
        let height = rgb_frame.height();

        let data = rgb_frame.data(0);

        let block_size = img_size;
        let step = (block_size as f64 / 4.0).round();
        let mut blocks = Vec::new();
        let mut blocks_images = Vec::new();

        for y in (0..height - block_size as u32).step_by(step as usize) {
            for x in (0..width - block_size as u32).step_by(step as usize) {

                let mut r = Vec::new();
                let mut g = Vec::new();
                let mut b = Vec::new();

                //used only to debug blocks images
                let mut block_pixels = Vec::new();

                for y1 in y..y+block_size as u32 {
                    for x1 in x..x+block_size as u32 {
                        let pixel_index = (y1 * (width*3) + (x1*3)) as usize;

                        r.push(data[pixel_index] as f32 / 255.0 * 10e-1);
                        g.push(data[pixel_index+1] as f32 / 255.0 * 10e-1);
                        b.push(data[pixel_index+2] as f32 / 255.0 * 10e-1);

                        /* Debug block images */
                        block_pixels.push(data[pixel_index]);
                        block_pixels.push(data[pixel_index+1]);
                        block_pixels.push(data[pixel_index+2]);
                        /* ****************** */
                    }
                }

                let mut input = Vec::new();
                input.push(Box::new(Tensor::from_data(block_size,block_size,r)));
                input.push(Box::new(Tensor::from_data(block_size,block_size,g)));
                input.push(Box::new(Tensor::from_data(block_size,block_size,b)));
                    
                blocks.push(input);

                /* Debug block images */
                let image_buffer = ImageBuffer::from_raw(block_size as u32, block_size as u32,  block_pixels.clone()).unwrap();
                let mut img = DynamicImage::ImageRgb8(image_buffer);
                //let block_image = img.as_rgb8().unwrap();
                //let mut debug_path = PathBuf::from("./debug");
                //let mut debug_filename = String::from("block_");
                //debug_filename.push_str(&format!("_{}.png", blocks.len()));
                //debug_path.push(debug_filename);
                //block_image.save(debug_path);
                /* ******************************* */
                blocks_images.push(img);
            }
        }

        /*let image_buffer = ImageBuffer::from_raw(width, height,  rgb_frame.data(0).to_vec()).unwrap();
        let mut img = DynamicImage::ImageRgb8(image_buffer);

        // get the original dimensions
        let width = img.width();
        let height = img.height();

        let image_size = (img_size*4) as u32;

        let new_width = if width > image_size { image_size } else { width };
        let new_height = if height > image_size {
            (height as f32 * new_width as f32 / width as f32) as u32
        } else {
            height
        };
        
        // resize the image
        let img = img.resize(new_width, new_height, image::imageops::FilterType::Lanczos3);

        let image = img.as_rgb8();
        if image.is_none() {
            return 0;
        }

        let image = image.unwrap();

        let block_size = img_size;
        let step = (block_size as f32 / 3.0).round();
        let mut blocks = Vec::new();
        
        // Iterate over the blocks of the image
        for y in (0..image.height() - block_size as u32).step_by(step as usize) {
            for x in (0..image.width() - block_size as u32).step_by(step as usize) {
                let mut sub_image = image.clone();

                let block = sub_image.sub_image(x as u32, y as u32, block_size as u32, block_size as u32);
                let block_image = block.to_image();

                let mut r = Vec::new();
                let mut g = Vec::new();
                let mut b = Vec::new();
        
                for row in block_image.chunks(block_size as usize * 3) {
                    for pixel in row.chunks_exact(3) {
                        r.push(pixel[0] as f32 / 255.0 * 10e-1);
                        g.push(pixel[1] as f32 / 255.0 * 10e-1);
                        b.push(pixel[2] as f32 / 255.0 * 10e-1);
                    }
                }
    
                let mut input = Vec::new();
                input.push(Box::new(Tensor::from_data(block_size,block_size,r)));
                input.push(Box::new(Tensor::from_data(block_size,block_size,g)));
                input.push(Box::new(Tensor::from_data(block_size,block_size,b)));
                    
                blocks.push(input);

                //let mut debug_path = PathBuf::from("./debug");
                //let mut debug_filename = String::from("block_");
                //debug_filename.push_str(&format!("_{}.png", blocks.len()));
                //debug_path.push(debug_filename);
                //
                //block_image.save(debug_path);
            }
        }*/

        let mut selected_blocks = Vec::new();

        let mut max_probability = 0.0;
        for (i,t) in blocks.iter().enumerate() {
            //let _ = std::io::stdout().flush();
            let output = neuron.forward(t.clone());
            if let Some(ref y) = output {
                for k in 0..y[0].cols() {
                    let value = y[0].get(0, k);
                    if value > 0.7 {
                        selected_blocks.push(i);
                    }
                    if value > 0.5 {
                        let mut debug_path = PathBuf::from("./debug");
                        let mut debug_filename = String::from("block_");
                        debug_filename.push_str(&format!("_{}_{}_{}.png", *idx.lock().unwrap(), i, value));
                        debug_path.push(debug_filename);
                        blocks_images.get(i).unwrap().save(debug_path);
                    }
                    if value > max_probability {
                        max_probability = value;
                    }
                }
            }    
        }

        if selected_blocks.len() > 0 {
            let mut debug_path = PathBuf::from("./debug");
            let mut debug_filename = idx.lock().unwrap().to_string();
            debug_filename.push_str(&format!("_{}.png", max_probability));
            debug_path.push(debug_filename);

            let mut wand = MagickWand::new();
            let image_data = [format!("P6\n{} {}\n255\n", dst_width, dst_height).as_bytes(),rgb_frame.data(0)].concat();
            if let Err(err) = wand.read_image_blob(image_data) {
                println!("Error reading image: {}", err);
            }

            for block in selected_blocks.iter() {

                let pos = *block as f64;
                let current_width = ((width as f64 - block_size as f64) / step).round();

                let y = (pos / current_width).round() * step;
                let x = (pos % current_width).round() * step;

                let original_x = (x / width as f64) * (dst_width as f64);
                let original_y = (y / height as f64) * (dst_height as f64);

                println!("Block {} at ({},{}) -> ({},{})",block,x,y,original_x,original_y);

                let mut drawing = DrawingWand::new();
                let mut pixel = PixelWand::new();
                pixel.set_color("#FF0000");
                drawing.set_stroke_color(&pixel);
                drawing.set_stroke_width(2.0);
                drawing.draw_rectangle(original_x as f64,original_y as f64,original_x+block_size as f64,original_y+block_size as f64);
                if let Err(error) = wand.draw_image(&drawing) {
                    println!("Error drawing image: {}", error);
                }
            }

            if let Err(error) = wand.write_image(debug_path.to_str().unwrap()) {
                println!("Error writing image: {}", error);
            }
        }

        *idx.lock().unwrap() += 1;

        let elapsed = start.elapsed();
        println!("Final Output={} in {:?} ms", max_probability,elapsed.as_millis());

        1
    }) {
        println!("Error connecting to camera: {}", err);
    }
}
