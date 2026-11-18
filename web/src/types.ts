export interface Character {
  id: string;
  name: string;
  gender: 'male' | 'female' | 'other';
  age_group: 'child' | 'young' | 'adult' | 'elderly';
  personality: string;
  aliases: string[];
  assigned_voice_id?: string;
  pca_x?: number;
  pca_y?: number;
}

export interface TurnStyle {
  emotion: string;
  intensity: number;
  pitch: 'low' | 'medium' | 'high';
  speed: 'slow' | 'medium' | 'fast';
  volume: 'low' | 'medium' | 'high';
  pause_before_ms: number;
  pause_after_ms: number;
  style_prompt?: string;
  reasoning?: string;
}

export interface Turn {
  idx: number;
  type: 'narration' | 'dialogue';
  speaker: string;
  text: string;
  addressed_to?: string | null;
  style: TurnStyle;
  audio_url?: string;
  duration?: number;
}

export interface VoiceProfile {
  id: string;
  name: string;
  gender: string;
  age_group: string;
  role: string;
  pitch_range: string;
  audio_path: string;
  audio_url?: string;
  pca_coords: [number, number];
}

export interface TimelineStem {
  turn_idx: number;
  speaker: string;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  emotion: string;
  text: string;
}

export interface StoryTimeline {
  master_audio_url: string;
  stems: TimelineStem[];
  total_duration: number;
  ambience_track?: string;
  lufs_level: number;
}
