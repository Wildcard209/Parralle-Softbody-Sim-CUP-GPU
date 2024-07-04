use nalgebra::Vector2;
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use rand::Rng;
use norman::normalize;
use norman::desc::Abs;

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub position: Vector2<f32>,
    pub prev_position: Vector2<f32>,
    pub velocity: Vector2<f32>,
    pub external_force: Vector2<f32>,
    pub mass: f32,
    pub damping_coeff: f32,
    pub adj_points: [Option<usize>; 4],
    pub has_physics: bool,
}

#[derive(Clone, Debug)]
pub struct Points {
    pub points: Vec<Point>,
    pub connections: Vec<(usize, usize)>,
}

impl Points {
    pub fn new(n: usize, m: usize) -> Self {
        Self {
            points: Vec::with_capacity(n * m),
            connections: Vec::new(),
        }
    }

    pub fn setup_points(&mut self, n: usize, m: usize, square_size: f32, initial_mass: f32, initial_damping_coeff: f32, defined_gap: bool, gap_size: f32) {
        self.initialize_points(n, m, initial_mass, initial_damping_coeff);
        self.calculate_positions(n, m, square_size , defined_gap, gap_size);
        self.set_adjacent_points(n, m);
        self.set_connections(n, m);
    }

    fn initialize_points(&mut self, n: usize, m: usize, initial_mass: f32, initial_damping_coeff: f32) {
        self.points.resize(n * m, Point {
            position: Vector2::new(0.0, 0.0),
            prev_position: Vector2::new(0.0, 0.0),
            velocity: Vector2::new(0.0, 0.0),
            external_force: Vector2::new(0.0, 0.0),
            mass: initial_mass,
            damping_coeff: initial_damping_coeff,
            adj_points: [None; 4],
            has_physics: true,
        });
    }

    fn calculate_positions(&mut self, n: usize, m: usize, square_size: f32, defined_gap: bool, gap_size: f32) {
        let gap_x;
        let gap_y;
        if defined_gap {
            gap_x = gap_size;
            gap_y = gap_size;
        }
        else {
            let total_gap_width = 2.0 - (n as f32 * square_size);
            let total_gap_height = 2.0 - (m as f32 * square_size);
            gap_x = total_gap_width / (n as f32 + 1.0);
            gap_y = total_gap_height / (m as f32 + 1.0);
        }
        

        for i in 0..m {
            for j in 0..n {
                let x0 = -1.0 + gap_x * (j as f32 + 1.0) + square_size * j as f32;
                let y0 = -0.5 + gap_y * (i as f32 + 1.0) + square_size * i as f32;

                let index = i * n + j;
                let point = &mut self.points[index];
                point.position = Vector2::new(x0, y0);
                point.prev_position = point.position;

                if (i == m - 1 && j == 0) || (i == m - 1 && j == n - 1) {
                    point.has_physics = false;
                }
            }
        }
    }

    fn set_adjacent_points(&mut self, n: usize, m: usize) {
        for i in 0..m {
            for j in 0..n {
                let index = i * n + j;
                let point = &mut self.points[index];
                let mut adj_index = 0;

                if i > 0 {
                    point.adj_points[adj_index] = Some((i - 1) * n + j);
                    adj_index += 1;
                }
                if i < m - 1 {
                    point.adj_points[adj_index] = Some((i + 1) * n + j);
                    adj_index += 1;
                }
                if j > 0 {
                    point.adj_points[adj_index] = Some(i * n + (j - 1));
                    adj_index += 1;
                }
                if j < n - 1 {
                    point.adj_points[adj_index] = Some(i * n + (j + 1));
                    adj_index += 1;
                }
                for k in adj_index..4 {
                    point.adj_points[k] = None;
                }
            }
        }
    }

    pub fn set_connections(&mut self, n: usize, m: usize) {
        for i in 0..m {
            for j in 0..n {
                let index = i * n + j;
                let point = &self.points[index];

                for k in 0..4 {
                    if let Some(adj_point_index) = point.adj_points[k] {
                        if index < adj_point_index {
                            self.connections.push((index, adj_point_index));
                        }
                    }
                }
            }
        }
    }

    pub fn update_external_forces(&mut self, x: f32, y: f32, distance_threshold: f32){
        for i in 0..self.points.len() {
            let point = &mut self.points[i];

            let dist = (x / 1000.0 - point.position.x) * (y / 1000.0 - point.position.y) + (y / 1000.0 - point.position.y) * (y / 1000.0 - point.position.y);

            if dist < distance_threshold
            {
                let mut movment_direction = Vector2::new((x / 1000.0) - point.position.x, (y / 1000.0) - point.position.y);
                normalize(&mut movment_direction.x, Abs::new());
                normalize(&mut movment_direction.y, Abs::new());
                point.external_force = movment_direction;
            }
        }
    }

    pub fn update_points(&mut self, num_of_threads: usize, gravity_enabled: bool, wind_enabled: bool,spring_coeff: f32, spring_relax_distance: f32, delta_time: f32) {
        let pool = ThreadPoolBuilder::new().num_threads(num_of_threads).build().unwrap();
    
        if wind_enabled {
            for i in 0..self.points.len() {
                let point = &mut self.points[i];
                point.external_force = Vector2::new(0.003, 0.0);
            }
        }


        let points_arc = Arc::new(Mutex::new(self.points.clone()));

        let forces: Vec<(Vector2<f32>, Vector2<f32>)> = pool.install(|| {
            (0..self.points.len())
                .into_par_iter()
                .map_init(|| points_arc.clone(), |points_arc, index| {
                    let points = points_arc.lock().unwrap();
                    let point = &points[index];
                    let mut force_x = 0.0;
                    let mut force_y = 0.0;
    
                    if point.has_physics {
                        for &adj_index in &point.adj_points {
                            if let Some(adj_index) = adj_index {
                                if let Some(adj_point) = points.get(adj_index) {
                                    let dx = adj_point.position.x - point.position.x;
                                    let dy = adj_point.position.y - point.position.y;
                                    let distance = ((dx * dx) + (dy * dy)).sqrt();
                                    let magnitude = spring_coeff * (distance - spring_relax_distance);
    
                                    force_x += magnitude * dx / distance;
                                    force_y += magnitude * dy / distance;
                                }
                            }
                        }
    
                        force_x += -point.velocity.x * point.damping_coeff;
                        force_y += -point.velocity.y * point.damping_coeff;
    
                        if gravity_enabled {
                            force_y += -9.81 * point.mass;
                        }
                        
                        let mut rng = rand::thread_rng();
                        if point.external_force.x != 0.0 
                        {
                            let random_number_x = rng.gen_range(0.0..=1.0);
                            //random_number_x = random_number_x * 2.0 - 1.0;
    
                            force_x += random_number_x * point.external_force.x;
                        }
                        
                        if point.external_force.y != 0.0 
                        {
                            let random_number_y = rng.gen_range(0.0..=1.0);
                            //random_number_y = random_number_y * 2.0 - 1.0;
    
                            force_y += random_number_y * point.external_force.y;
                        }
                    }
    
                    (Vector2::new(force_x, force_y), point.position)
                })
                .collect()
        });
    
        let new_points: Vec<Point> = pool.install(|| {
            (0..self.points.len())
                .into_par_iter()
                .map_init(|| points_arc.clone(), |points_arc, index| {
                    let points = points_arc.lock().unwrap();
                    let point = &points[index];
                    let mut new_point = point.clone();
    
                    if point.has_physics {
                        let (force, _prev_position) = &forces[index];
                        let accel = force / point.mass;
    
                        let new_pos_x = new_point.position.x + new_point.velocity.x * delta_time + 0.5 * accel.x * (delta_time * delta_time);
                        let new_pos_y = new_point.position.y + new_point.velocity.y * delta_time + 0.5 * accel.y * (delta_time * delta_time);
    
                        new_point.prev_position = point.position;
                        new_point.position.x = new_pos_x;
                        new_point.position.y = new_pos_y;
    
                        new_point.velocity.x = (new_point.position.x - new_point.prev_position.x) / delta_time;
                        new_point.velocity.y = (new_point.position.y - new_point.prev_position.y) / delta_time;
    
                        new_point.external_force = Vector2::new(0.0, 0.0);
                    }
    
                    new_point
                })
                .collect()
        });
    
        self.points = new_points;
    }
}
