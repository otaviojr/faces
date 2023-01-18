use std::{sync::Mutex, fs::{self, DirEntry}, str::FromStr, error::Error, io::Write, path::PathBuf};
use image::{GenericImage, EncodableLayout, GenericImageView, imageops};
use rand::thread_rng;
use rand::seq::SliceRandom;
use neuron::{Neuron, math::Tensor, layers::{LinearLayer, LinearLayerConfig, ConvLayer, ConvLayerConfig, FlattenLayer, PoolingLayer, PoolingLayerConfig}, activations::{ReLU, Sigmoid, Tanh, SoftMax}, cost::Functions, pipeline::SequentialPieline, Loader};

fn main() {

    let mut neuron = Neuron::new();
    let mut pipeline = SequentialPieline::new();

    let img_size: usize = 24;

/*
    let mut input = Vec::new();
    for i in 0..3 {
        input.push(Box::new(Tensor::random(10, 10, -1.0 as f64, 1.0 as f64)))
    }

    neuron.add_layer(Box::new(Mutex::new(ConvLayer::new(3, 10, (3,3),ConvLayerConfig { activation: Box::new(ReLU::new()), learn_rate: 0.00001, padding: 0, stride: 1, weights_low: (-2.0 as f64).exp(), weights_high: (2.0 as f64).exp()}))));

    let output = neuron.forward(input.clone());
    if let Some(ref y) = output {
        println!("Starting Backward Propagation");
        println!("Input = {}",input[0]);
        println!("output = {}",neuron.backward(y.clone()).unwrap()[0]);
    }
    return;
*/
    /*
    neuron.add_layer(Box::new(Mutex::new(FlattenLayer::new())));
    neuron.add_layer(Box::new(Mutex::new(LinearLayer::new(128*128*3, 100, LinearLayerConfig{ activation: Box::new(ReLU::new()), learn_rate: 0.01}))));
    neuron.add_layer(Box::new(Mutex::new(LinearLayer::new(100, 50, LinearLayerConfig{ activation: Box::new(ReLU::new()), learn_rate: 0.01}))));
    neuron.add_layer(Box::new(Mutex::new(LinearLayer::new(50, 1, LinearLayerConfig{ activation: Box::new(Sigmoid::new()), learn_rate: 0.01}))));
    */
    
    
    pipeline.add_layer(Mutex::new(Box::new(ConvLayer::new("conv1".to_owned(), 3, 14, (5,5), ConvLayerConfig { activation: Box::new(ReLU::new()), learn_rate: 10e-4, padding: 0, stride: 1 }))));
    pipeline.add_layer(Mutex::new(Box::new(PoolingLayer::new((2,2), PoolingLayerConfig { stride: 2}))));
    //neuron.add_layer(Mutex::new(Box::new(ConvLayer::new(14, 36, (5,5), ConvLayerConfig { activation: Box::new(ReLU::new()), learn_rate: 10e-6, padding: 0, stride: 1, weights_low: -1.5 * (-10.0 as f64).exp(), weights_high:  1.5 * (-10.0 as f64).exp()}))));
    //neuron.add_layer(Mutex::new(Box::new(PoolingLayer::new((2,2), PoolingLayerConfig { stride: 2 }))));
    //neuron.add_layer(Mutex::new(Box::new(ConvLayer::new(36, 36, (3,3), ConvLayerConfig { activation: Box::new(ReLU::new()), learn_rate: 10e-7, padding: 0, stride: 1, weights_low: -1.5 * (-10.0 as f64).exp(), weights_high:  1.5 * (-10.0 as f64).exp()}))));
    //neuron.add_layer(Mutex::new(Box::new(PoolingLayer::new((2,2), PoolingLayerConfig { stride: 1 }))));
    //neuron.add_layer(Mutex::new(Box::new(ConvLayer::new(36, 16, (3,3), ConvLayerConfig { activation: Box::new(ReLU::new()), learn_rate: 10e-7, padding: 0, stride: 1, weights_low: -1.5 * (-10.0 as f64).exp(), weights_high:  1.5 * (-10.0 as f64).exp()}))));
    //neuron.add_layer(Mutex::new(Box::new(PoolingLayer::new((2,2), PoolingLayerConfig { stride: 2 }))));
    pipeline.add_layer(Mutex::new(Box::new(FlattenLayer::new())));
    pipeline.add_layer(Mutex::new(Box::new(LinearLayer::new("lin1".to_owned(),1400, 1400, LinearLayerConfig{ activation: Box::new(ReLU::new()), learn_rate: 10e-4}))));
    //neuron.add_layer(Mutex::new(Box::new(LinearLayer::new(144, 64, LinearLayerConfig{ activation: Box::new(ReLU::new()), learn_rate: 10e-8, weights_low: -1.5 * (-10.0 as f64).exp(), weights_high: 1.5 * (-10.0 as f64).exp()}))));
    pipeline.add_layer(Mutex::new(Box::new(LinearLayer::new("lin2".to_owned(),1400, 2, LinearLayerConfig{ activation: Box::new(SoftMax::new()), learn_rate: 10e-4}))));

    neuron.add_pipeline(Mutex::new(Box::new(pipeline)));

    if let Err(error) = neuron.load("faces.weigths") {
        println!("Error load weigths: ${}", error);
    }

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
                    r.push(pixel[0] as f64 / 255.0 * 10e-1);
                    g.push(pixel[1] as f64 / 255.0 * 10e-1);
                    b.push(pixel[2] as f64 / 255.0 * 10e-1);
                }
            }

            let mut input = Vec::new();
            input.push(Box::new(Tensor::from_data(img_size,img_size,r)));
            input.push(Box::new(Tensor::from_data(img_size,img_size,g)));
            input.push(Box::new(Tensor::from_data(img_size,img_size,b)));
                
            let label_value = filename_string.to_str().unwrap().chars().nth(0).unwrap();
            let label_digit = label_value.to_digit(0x10).unwrap() as f64;
    
            println!("Starting Forward Propagation");
            let output = neuron.forward(input.clone());
            if let Some(ref y) = output {
                println!("Output Size={}x{}", y[0].rows(), y[0].cols());
                println!("Output={:?} - {} - {} - {}", y[0], idx, label_digit, filename_string.to_string_lossy());
                println!("Cost: {:?} - {} - {}", Functions::binary_cross_entropy_loss(&y[0], &Tensor::from_data(1,1,vec![label_digit])), idx, filename_string.to_string_lossy());
                println!("Starting Backward Propagation");
                //let loss = Functions::binary_cross_entropy_loss_derivative(&y[0], &Tensor::from_data(2,1,vec![if label_digit == 1.0 {1.0} else {y[0].get(0,0)}, if label_digit == 0.0 {1.0} else {y[0].get(1,0)}]));
                let loss = y[0].sub(&Tensor::from_data(2,1,vec![if label_digit == 1.0 {1.0} else {0.0}, if label_digit == 0.0 {1.0} else {0.0}]));
                //let loss = Functions::softmax_loss(&y[0],&Tensor::from_data(2,1,vec![if label_digit == 1.0 {1.0} else {0.0}, if label_digit == 0.0 {1.0} else {0.0}]));
                println!("Loss: {:?}", loss);
                //let loss = Functions::binary_cross_entropy_loss_derivative(&y[0], &Tensor::from_data(1,1,vec![label_digit]));
                //let loss = y[0].sub(&Tensor::from_data(1,1,vec![label_digit]));
                //if label_digit == 1.0 {
                    neuron.backward(vec![Box::new(loss)]);
                //}
            }

            if idx > 0 && idx % 100 == 0 {
                if let Err(error) = neuron.save("faces.weigths") {
                    println!("Error saving weigths: ${}", error);
                }
            }

            //move file to final_dir
            let mut final_path = PathBuf::from(final_dir);
            final_path.push(filename_string);
            let _ = fs::rename(p, final_path);
            
        }
    }

    let dir = "./test/";
    let mut files = Vec::new();
    let dir = fs::read_dir(dir).unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        let path = entry.path();
        files.push(path);
    }

    for path in files.iter() {
        let filename = path.file_name().unwrap();
        let filename_string = filename.to_os_string();
        println!("{}", filename_string.to_string_lossy());
        let p = path.as_path();
        let mut img = image::open(p).unwrap();
        img = img.resize_to_fill((img_size*4) as u32, (img_size*4) as u32, imageops::FilterType::Nearest);

        
        let image = img.as_rgb8();
        if image.is_none() {
            let _ = fs::remove_file(p);
            continue;
        }

        let image = image.unwrap();

        let block_size = img_size;
        let mut blocks = Vec::new();

        // Iterate over the blocks of the image
        for y in (0..image.height()).step_by(block_size) {
            for x in (0..image.width()).step_by(block_size) {
                let mut sub_image = image.clone();
                let block = sub_image.sub_image(x as u32, y as u32, block_size as u32, block_size as u32);
                let block_image = block.to_image();

                let mut r = Vec::new();
                let mut g = Vec::new();
                let mut b = Vec::new();
        
                for row in block_image.chunks(block_size as usize * 3) {
                    for pixel in row.chunks_exact(3) {
                        r.push(pixel[0] as f64 / 255.0 * 10e-1);
                        g.push(pixel[1] as f64 / 255.0 * 10e-1);
                        b.push(pixel[2] as f64 / 255.0 * 10e-1);
                    }
                }
    
                let mut input = Vec::new();
                input.push(Box::new(Tensor::from_data(block_size,block_size,r)));
                input.push(Box::new(Tensor::from_data(block_size,block_size,g)));
                input.push(Box::new(Tensor::from_data(block_size,block_size,b)));
                    
                blocks.push(input);
            }
        }

        let mut max_probability = 0.0;
        for (i,t) in blocks.iter().enumerate() {
            println!("Processing {}... {}/{} ", filename_string.to_string_lossy(), i, blocks.len());
            //let _ = std::io::stdout().flush();
            let output = neuron.forward(t.clone());
            if let Some(ref y) = output {
                for i in 0..y[0].cols() {
                    if y[0].get(0, i) > max_probability {
                        max_probability = y[0].get(0, i);
                    }
                }
            }    
        }
        println!("Final Output={}={}", filename_string.to_string_lossy(), max_probability);
    }
}
