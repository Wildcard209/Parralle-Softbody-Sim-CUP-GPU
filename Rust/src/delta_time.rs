use std::time::{Duration, Instant};

pub struct DeltaTime {
    last_frame_time: Instant,
}

impl DeltaTime {
    pub fn new() -> Self {
        Self {
            last_frame_time: Instant::now(),
        }
    }

    fn update(&mut self) -> Duration {
        let now = Instant::now();
        let delta_time = now.duration_since(self.last_frame_time);
        self.last_frame_time = now;
        delta_time
    }

    pub fn update_secs(&mut self) -> f32 {
        self.update().as_secs_f32()
    }
}