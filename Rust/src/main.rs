#[macro_use]
extern crate glium;
extern crate winit;
use winit::event::ElementState;
use winit::keyboard::Key;
use crate::winit::platform::modifier_supplement::KeyEventExtModifierSupplement;

use num_cpus;

mod points;
use points::Points;

mod vertices;
use vertices::Vertices;

mod vertex;
use vertex::Vertex;

mod delta_time;
use delta_time::DeltaTime;

use std::time::Instant;

fn main() {
    //Define consts
    let n = 20;
    let m = 20;
    let square_size = 0.01;
    let initial_mass = 0.0004;
    let initial_damping_coeff = 0.003;
    let spring_coeff = 4.0;
    

    let spring_relax_distance = 0.0956;

    //10 x 10 distance
    //let spring_relax_distance = 0.183;
    let distance_threshold = 20.0;

    let define_gap = true;
    let gap_size = 0.08571428;

    let mut gravity_enabled = false;
    let mut visuals_enabled = false;
    let mut wind_enabled = false;
    let test_enabled = false;

    let core_tread_multiplier = 2;

    let mut last_poke = Instant::now();
    let timer = Instant::now();
    let mut run_counter = 0;

    //Set up points

    let mut points = Points::new(n, m);
    points.setup_points(n, m, square_size, initial_mass, initial_damping_coeff, define_gap, gap_size);
    //Get core count

    let num_of_threads = num_cpus::get() * core_tread_multiplier;
    println!("The number of threads used is {:}, the current cpu core count is {:}, and the core thread multiplier is {:}", num_of_threads, num_cpus::get(), core_tread_multiplier);

    //Start open GL

    #[allow(unused_imports)]
    use glium::{glutin, Surface};

    let event_loop = winit::event_loop::EventLoopBuilder::new().build().expect("event loop building");
    let (_window, display) = glium::backend::glutin::SimpleWindowBuilder::new().build(&event_loop);

    let mut vertices = Vertices::new();

    implement_vertex!(Vertex, position);

    vertices.setup_vertices(&points, n, m, square_size);

    let mut _vertex_buffer = glium::VertexBuffer::new(&display, &vertices.data).unwrap();
    let indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);

    let vertex_shader_src = r#"
        #version 140

        in vec2 position;

        uniform mat4 matrix;

        void main() {
            gl_Position = matrix * vec4(position, 0.0, 1.0);
        }
    "#;

    let fragment_shader_src = r#"
        #version 140

        out vec4 color;

        void main() {
            color = vec4(0.0, 1.0, 0.0, 1.0);
        }
    "#;

    let program = glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None).unwrap();

    let mut delta_time = DeltaTime::new();

    let _ = event_loop.run(move |event, window_target| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => match event {
                winit::event::WindowEvent::CloseRequested => window_target.exit(),

                winit::event::WindowEvent::Resized(window_size) => {
                    display.resize(window_size.into());
                },

                winit::event::WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed && !event.repeat {
                        match event.key_without_modifiers().as_ref() {
                            Key::Character("g") => {
                                gravity_enabled = !gravity_enabled;
                                },
                            Key::Character("v") => {
                                visuals_enabled = !visuals_enabled;
                                },
                            Key::Character("w") => {
                                wind_enabled = !wind_enabled;
                            }
                                _ => (),
                            }
                    }
                },

                winit::event::WindowEvent::CursorMoved { position, .. } => {
                    let time_since_last_poke = Instant::now().duration_since(last_poke).as_secs_f32();

                    if time_since_last_poke > 2.0 {
                    last_poke = Instant::now();
                    points.update_external_forces(position.x as f32, position.y as f32, distance_threshold);
                    }
                },

                winit::event::WindowEvent::RedrawRequested => {

                    let next_frame_time = std::time::Instant::now() + std::time::Duration::from_nanos(16_666_667);
                    winit::event_loop::ControlFlow::WaitUntil(next_frame_time);
                                  
                    for _i in 0 .. 10 {
                        let mut delta_t = format!("{:.4}",delta_time.update_secs()).parse().unwrap();

                        if delta_t > 0.004 {
                            delta_t = 0.004;
                        }

                        if delta_t < 0.0 {
                            delta_t = 0.005;
                        }

                        points.update_points(num_of_threads, gravity_enabled, wind_enabled, spring_coeff, spring_relax_distance, delta_t);
                        if test_enabled
                        {
                            run_counter += 1;
                            let elapsed_secs = timer.elapsed().as_secs();

                            if elapsed_secs != 0 && run_counter != 0
                            {
                                let calculations_per_sec = run_counter / elapsed_secs;

                                println!("{:}", calculations_per_sec);
                            }
                        }
                    }

                    let mut target = display.draw();

                    target.clear_color(0.0, 0.0, 0.0, 1.0);

                    if visuals_enabled
                    {
                        vertices.update_vertices(points.clone(), square_size);

                        _vertex_buffer = glium::VertexBuffer::new(&display, &vertices.data).unwrap();

                    
                        let scale_x: f32 = 0.4;
                        let scale_y: f32 = 0.7;

                        let pos_x : f32 = 0.0;
                        let pos_y : f32 = 0.0;
                        let pos_z : f32 = 0.0;
                        
                        let uniforms = uniform! {
                            matrix: [
                                [scale_x, 0.0, 0.0, 0.0],
                                [0.0, scale_y, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [pos_x, pos_y, pos_z, 1.0],
                            ]
                        };
                        target.draw(&_vertex_buffer, &indices, &program, &uniforms, &Default::default()).unwrap();
                    }
                    
                    target.finish().unwrap();

                    // End render loop
                },
                _ => (),
            },            
            winit::event::Event::AboutToWait => {
                _window.request_redraw();
            },
            _ => (),
        };
    });
}
