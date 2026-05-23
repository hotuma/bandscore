/**
 * BeatNet-Plus API クライアント
 *
 * ファインチューニング、モデル管理、データセット管理のための API 関数
 */

export interface ModelInfo {
  model_id: string;
  name: string;
  created_at: string;
  base_model_id?: string;
  training_dataset_id?: string;
  is_default: boolean;
}

export interface DatasetInfo {
  dataset_id: string;
  name: string;
  description: string;
  audio_files: string[];
  created_at: string;
  file_count: number;
}

export interface TrainingJob {
  job_id: string;
  job_name: string;
  status: 'queued' | 'running' | 'completed' | 'error';
  progress: number;
  epoch: number;
  total_epochs: number;
  loss?: number;
  started_at?: string;
  completed_at?: string;
  error?: string;
  dataset_id: string;
  model_id: string;
  model_id_result?: string;
}

const BEATNET_API_URL = process.env.NEXT_PUBLIC_BEATNET_API_URL || 'http://localhost:8001';
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000';

// ============================================================================
// モデル管理
// ============================================================================

export async function listModels(): Promise<ModelInfo[]> {
  const response = await fetch(`${BEATNET_API_URL}/models`);
  if (!response.ok) throw new Error('Failed to fetch models');
  return response.json();
}

export async function getModelInfo(modelId: string): Promise<any> {
  const response = await fetch(`${BEATNET_API_URL}/models/${modelId}`);
  if (!response.ok) throw new Error('Failed to fetch model info');
  return response.json();
}

export async function selectModel(modelId: string): Promise<any> {
  const response = await fetch(`${API_BASE_URL}/beatnet/select-model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_id: modelId }),
  });
  if (!response.ok) throw new Error('Failed to select model');
  return response.json();
}

export async function deleteModel(modelId: string): Promise<any> {
  const response = await fetch(`${BEATNET_API_URL}/models/${modelId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete model');
  return response.json();
}

// ============================================================================
// データセット管理
// ============================================================================

export async function createDataset(name: string, description?: string): Promise<DatasetInfo> {
  const response = await fetch(`${BEATNET_API_URL}/datasets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, description: description || '' }),
  });
  if (!response.ok) throw new Error('Failed to create dataset');
  return response.json();
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  const response = await fetch(`${BEATNET_API_URL}/datasets`);
  if (!response.ok) throw new Error('Failed to fetch datasets');
  return response.json();
}

export async function uploadToDataset(
  datasetId: string,
  audioFile: File,
  beatsFile: File
): Promise<any> {
  const formData = new FormData();
  formData.append('audio_file', audioFile);
  formData.append('beats_file', beatsFile);

  const response = await fetch(`${BEATNET_API_URL}/datasets/${datasetId}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!response.ok) throw new Error('Failed to upload to dataset');
  return response.json();
}

// ============================================================================
// トレーニング管理
// ============================================================================

export async function startTraining(params: {
  job_name: string;
  model_id?: string;
  dataset_id: string;
  epochs?: number;
  batch_size?: number;
  learning_rate?: number;
}): Promise<{ job_id: string; status: string; message: string }> {
  const response = await fetch(`${BEATNET_API_URL}/training/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  if (!response.ok) throw new Error('Failed to start training');
  return response.json();
}

export async function getTrainingStatus(jobId: string): Promise<TrainingJob> {
  const response = await fetch(`${BEATNET_API_URL}/training/${jobId}`);
  if (!response.ok) throw new Error('Failed to fetch training status');
  return response.json();
}

export async function listTrainingJobs(): Promise<TrainingJob[]> {
  const response = await fetch(`${BEATNET_API_URL}/training/jobs`);
  if (!response.ok) throw new Error('Failed to fetch training jobs');
  return response.json();
}

// ============================================================================
// ヘルスチェック
// ============================================================================

export async function checkHealth(): Promise<{ status: string; service: string; model_loaded: boolean }> {
  const response = await fetch(`${BEATNET_API_URL}/health`);
  if (!response.ok) throw new Error('Health check failed');
  return response.json();
}
