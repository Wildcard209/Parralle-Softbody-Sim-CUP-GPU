use crate::vertex::Vertex;
use crate::points::Points;

pub struct Vertices {
    vertices_points: Vec<Vertex>,
    vertices_connections: Vec<Vertex>,
    pub data : Vec<Vertex>,
}

impl Vertices {
    pub fn new() -> Self {
        Vertices { 
            vertices_points: Vec::new(),
            vertices_connections: Vec::new(),
            data: Vec::new() 
        }
    }

    pub fn setup_vertices(&mut self, points: &Points, n: usize, m: usize, square_size: f32) {
        self.initialize_points(n, m, square_size, &points);
        self.initialize_connections(&points, square_size);
        self.calculate_data();
    }

    fn initialize_points(&mut self, n: usize, m: usize, square_size: f32, points: &Points) {
        let num_vertices_points = n * m * 6;
        self.vertices_points.resize(num_vertices_points, Vertex { position: [0.0, 0.0] });

        let mut vertex_index_points = 0;

        for i in 0..m {
            for j in 0..n {
                let x0 = points.points[i * n + j].position.x;
                let y0 = points.points[i * n + j].position.y;

                let x1 = x0 + square_size;
                let y1 = y0 + square_size;

                self.vertices_points[vertex_index_points] = Vertex { position: [x0, y0] }; vertex_index_points += 1;
                self.vertices_points[vertex_index_points] = Vertex { position: [x1, y0] }; vertex_index_points += 1;
                self.vertices_points[vertex_index_points] = Vertex { position: [x0, y1] }; vertex_index_points += 1;

                self.vertices_points[vertex_index_points] = Vertex { position: [x0, y1] }; vertex_index_points += 1;
                self.vertices_points[vertex_index_points] = Vertex { position: [x1, y0] }; vertex_index_points += 1;
                self.vertices_points[vertex_index_points] = Vertex { position: [x1, y1] }; vertex_index_points += 1;
            }
        }
    }

    fn initialize_connections(&mut self, points: &Points, square_size: f32) {
        let num_vertices_connections = points.connections.len() * 6;
        self.vertices_connections.resize(num_vertices_connections, Vertex { position: [0.0, 0.0] });
    
        let mut vertex_index_connections = 0;
    
        for (connection_one, connection_two) in &points.connections {
            let point_one = &points.points[*connection_one];
            let point_two = &points.points[*connection_two];
    
            let dx = point_two.position.x - point_one.position.x;
            let dy = point_two.position.y - point_one.position.y;
            let distance = (dx * dx + dy * dy).sqrt();
    
            let unit_x = dx / distance;
            let unit_y = dy / distance;
    
            let perp_x = -unit_y;
            let perp_y = unit_x;
    
            let offset_x = perp_x * square_size / 2.0;
            let offset_y = perp_y * square_size / 2.0;

            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_one.position.x - offset_x, point_one.position.y - offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_one.position.x + offset_x, point_one.position.y + offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_two.position.x + offset_x, point_two.position.y + offset_y] }; vertex_index_connections += 1;
    
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_one.position.x - offset_x, point_one.position.y - offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_two.position.x + offset_x, point_two.position.y + offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_two.position.x - offset_x, point_two.position.y - offset_y] }; vertex_index_connections += 1;
        }
    }

    fn calculate_data(&mut self) {
        self.data.clear();
        self.data.extend(&self.vertices_points);
        self.data.extend(&self.vertices_connections);
    }

    pub fn update_vertices(&mut self, points: Points, square_size: f32) {
        let mut vertex_index_points = 0;

        for (_index, point) in points.points.iter().enumerate() {
            let x0 = point.position.x;
            let y0 = point.position.y;

            let x1 = x0 + square_size;
            let y1 = y0 + square_size;

            self.vertices_points[vertex_index_points] = Vertex { position: [x0, y0] }; vertex_index_points += 1;
            self.vertices_points[vertex_index_points] = Vertex { position: [x1, y0] }; vertex_index_points += 1;
            self.vertices_points[vertex_index_points] = Vertex { position: [x0, y1] }; vertex_index_points += 1;

            self.vertices_points[vertex_index_points] = Vertex { position: [x0, y1] }; vertex_index_points += 1;
            self.vertices_points[vertex_index_points] = Vertex { position: [x1, y0] }; vertex_index_points += 1;
            self.vertices_points[vertex_index_points] = Vertex { position: [x1, y1] }; vertex_index_points += 1;
        }

        let mut vertex_index_connections = 0;

        for (connection_one, connection_two) in &points.connections {
            let point_one = &points.points[*connection_one];
            let point_two = &points.points[*connection_two];

            let dx = point_two.position.x - point_one.position.x;
            let dy = point_two.position.y - point_one.position.y;
            let distance = (dx * dx + dy * dy).sqrt();

            let unit_x = dx / distance;
            let unit_y = dy / distance;

            let perp_x = -unit_y;
            let perp_y = unit_x;

            let offset_x = perp_x * square_size / 2.0;
            let offset_y = perp_y * square_size / 2.0;

            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_one.position.x - offset_x, point_one.position.y - offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_one.position.x + offset_x, point_one.position.y + offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_two.position.x + offset_x, point_two.position.y + offset_y] }; vertex_index_connections += 1;
        
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_one.position.x - offset_x, point_one.position.y - offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_two.position.x + offset_x, point_two.position.y + offset_y] }; vertex_index_connections += 1;
            self.vertices_connections[vertex_index_connections] = Vertex { position: [point_two.position.x - offset_x, point_two.position.y - offset_y] }; vertex_index_connections += 1;
        }

        self.calculate_data();
    }
}