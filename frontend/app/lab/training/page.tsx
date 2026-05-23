'use client';

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import {
  listModels,
  listDatasets,
  startTraining,
  getTrainingStatus,
  listTrainingJobs,
  TrainingJob,
  ModelInfo,
  DatasetInfo,
  createDataset,
  uploadToDataset,
  deleteModel
} from '../../../lib/beatnetApi';

export default function TrainingPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [activeTab, setActiveTab] = useState<'models' | 'datasets' | 'training'>('training');

  // Training form state
  const [jobName, setJobName] = useState('');
  const [selectedModel, setSelectedModel] = useState('default');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [epochs, setEpochs] = useState(10);
  const [isTraining, setIsTraining] = useState(false);

  // Dataset creation state
  const [showCreateDataset, setShowCreateDataset] = useState(false);
  const [newDatasetName, setNewDatasetName] = useState('');
  const [newDatasetDesc, setNewDatasetDesc] = useState('');
  const [uploadingDatasetId, setUploadingDatasetId] = useState<string | null>(null);

  // Upload state
  const [uploadAudioFile, setUploadAudioFile] = useState<File | null>(null);
  const [uploadBeatsFile, setUploadBeatsFile] = useState<File | null>(null);
  const [uploadTargetDataset, setUploadTargetDataset] = useState<string | null>(null);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadJobs, 3000); // Poll job status every 3s
    return () => clearInterval(interval);
  }, []);

  const loadData = async () => {
    try {
      const [modelsRes, datasetsRes] = await Promise.all([
        listModels(),
        listDatasets()
      ]);
      setModels(modelsRes);
      setDatasets(datasetsRes);
    } catch (error) {
      console.error('Failed to load data:', error);
    }
  };

  const loadJobs = async () => {
    try {
      const jobsRes = await listTrainingJobs();
      setJobs(jobsRes);
    } catch (error) {
      console.error('Failed to load jobs:', error);
    }
  };

  const handleStartTraining = async () => {
    if (!jobName || !selectedDataset) {
      alert('Please fill in all required fields');
      return;
    }

    setIsTraining(true);
    try {
      const result = await startTraining({
        job_name: jobName,
        model_id: selectedModel,
        dataset_id: selectedDataset,
        epochs,
      });
      alert(`Training started: ${result.job_id}`);
      setJobName('');
      loadJobs();
    } catch (error) {
      alert(`Failed to start training: ${error}`);
    } finally {
      setIsTraining(false);
    }
  };

  const handleCreateDataset = async () => {
    if (!newDatasetName) {
      alert('Please enter a dataset name');
      return;
    }

    try {
      const dataset = await createDataset(newDatasetName, newDatasetDesc);
      setDatasets([...datasets, dataset]);
      setNewDatasetName('');
      setNewDatasetDesc('');
      setShowCreateDataset(false);
      alert('Dataset created successfully');
    } catch (error) {
      alert(`Failed to create dataset: ${error}`);
    }
  };

  const handleDatasetUpload = async (datasetId: string, audioFile: File, beatsFile: File) => {
    setUploadingDatasetId(datasetId);
    try {
      await uploadToDataset(datasetId, audioFile, beatsFile);
      alert('Files uploaded successfully');
      loadData();
      setUploadAudioFile(null);
      setUploadBeatsFile(null);
      setUploadTargetDataset(null);
    } catch (error) {
      alert(`Failed to upload: ${error}`);
    } finally {
      setUploadingDatasetId(null);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    if (modelId === 'default') return;
    if (!confirm(`Delete model ${modelId}?`)) return;

    try {
      await deleteModel(modelId);
      setModels(models.filter(m => m.model_id !== modelId));
    } catch (error) {
      alert(`Failed to delete model: ${error}`);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="mb-6">
          <Link href="/lab" className="text-blue-600 hover:text-blue-800">
            ← Back to Lab
          </Link>
        </div>
        <h1 className="text-3xl font-bold mb-8">BeatNet-Plus Training</h1>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-6">
          {(['models', 'datasets', 'training'] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg capitalize ${
                activeTab === tab
                  ? 'bg-blue-600 text-white'
                  : 'bg-white text-gray-700 hover:bg-gray-100'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Models Tab */}
        {activeTab === 'models' && (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Available Models</h2>
            <div className="space-y-3">
              {models.map((model) => (
                <div
                  key={model.model_id}
                  className="flex items-center justify-between p-4 border rounded-lg"
                >
                  <div>
                    <div className="font-medium">{model.name}</div>
                    <div className="text-sm text-gray-500">
                      ID: {model.model_id}
                      {model.is_default && ' (Default)'}
                    </div>
                    {model.created_at && (
                      <div className="text-xs text-gray-400">Created: {model.created_at}</div>
                    )}
                  </div>
                  {!model.is_default && (
                    <button
                      onClick={() => handleDeleteModel(model.model_id)}
                      className="px-3 py-1 text-sm text-red-600 hover:bg-red-50 rounded"
                    >
                      Delete
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Datasets Tab */}
        {activeTab === 'datasets' && (
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Training Datasets</h2>
              <button
                onClick={() => setShowCreateDataset(!showCreateDataset)}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Create Dataset
              </button>
            </div>

            {showCreateDataset && (
              <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                <h3 className="font-medium mb-3">New Dataset</h3>
                <input
                  type="text"
                  placeholder="Dataset Name"
                  value={newDatasetName}
                  onChange={(e) => setNewDatasetName(e.target.value)}
                  className="w-full mb-2 px-3 py-2 border rounded"
                />
                <input
                  type="text"
                  placeholder="Description (optional)"
                  value={newDatasetDesc}
                  onChange={(e) => setNewDatasetDesc(e.target.value)}
                  className="w-full mb-3 px-3 py-2 border rounded"
                />
                <button
                  onClick={handleCreateDataset}
                  className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
                >
                  Create
                </button>
                <button
                  onClick={() => setShowCreateDataset(false)}
                  className="ml-2 px-4 py-2 bg-gray-300 rounded hover:bg-gray-400"
                >
                  Cancel
                </button>
              </div>
            )}

            <div className="space-y-4">
              {datasets.map((dataset) => (
                <DatasetCard
                  key={dataset.dataset_id}
                  dataset={dataset}
                  onUpload={(audio, beats) => handleDatasetUpload(dataset.dataset_id, audio, beats)}
                  isUploading={uploadingDatasetId === dataset.dataset_id}
                  uploadAudioFile={uploadAudioFile}
                  uploadBeatsFile={uploadBeatsFile}
                  uploadTargetDataset={uploadTargetDataset}
                  setUploadAudioFile={setUploadAudioFile}
                  setUploadBeatsFile={setUploadBeatsFile}
                  setUploadTargetDataset={setUploadTargetDataset}
                />
              ))}
            </div>
          </div>
        )}

        {/* Training Tab */}
        {activeTab === 'training' && (
          <div className="space-y-6">
            {/* Training Form */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Start Training</h2>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Job Name</label>
                  <input
                    type="text"
                    value={jobName}
                    onChange={(e) => setJobName(e.target.value)}
                    className="w-full px-3 py-2 border rounded"
                    placeholder="e.g., My Guitar Style Model"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Base Model</label>
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full px-3 py-2 border rounded"
                  >
                    {models.map((model) => (
                      <option key={model.model_id} value={model.model_id}>
                        {model.name} {model.is_default ? '(Default)' : ''}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Dataset</label>
                  <select
                    value={selectedDataset}
                    onChange={(e) => setSelectedDataset(e.target.value)}
                    className="w-full px-3 py-2 border rounded"
                  >
                    <option value="">Select a dataset</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.dataset_id} value={dataset.dataset_id}>
                        {dataset.name} ({dataset.file_count} files)
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Epochs</label>
                  <input
                    type="number"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                    className="w-full px-3 py-2 border rounded"
                    min={1}
                    max={100}
                  />
                </div>
              </div>
              <button
                onClick={handleStartTraining}
                disabled={isTraining || !jobName || !selectedDataset}
                className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {isTraining ? 'Starting...' : 'Start Training'}
              </button>
            </div>

            {/* Training Jobs */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Training Jobs</h2>
              <div className="space-y-3">
                {jobs.length === 0 ? (
                  <p className="text-gray-500">No training jobs yet</p>
                ) : (
                  jobs.map((job) => (
                    <TrainingJobCard key={job.job_id} job={job} />
                  ))
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Helper Components
function DatasetCard({
  dataset,
  onUpload,
  isUploading,
  uploadAudioFile,
  uploadBeatsFile,
  uploadTargetDataset,
  setUploadAudioFile,
  setUploadBeatsFile,
  setUploadTargetDataset
}: {
  dataset: DatasetInfo;
  onUpload: (audio: File, beats: File) => void;
  isUploading: boolean;
  uploadAudioFile: File | null;
  uploadBeatsFile: File | null;
  uploadTargetDataset: string | null;
  setUploadAudioFile: (file: File | null) => void;
  setUploadBeatsFile: (file: File | null) => void;
  setUploadTargetDataset: (id: string | null) => void;
}) {
  const isExpanded = uploadTargetDataset === dataset.dataset_id;

  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="font-medium">{dataset.name}</h3>
          <p className="text-sm text-gray-500">{dataset.description}</p>
          <div className="text-xs text-gray-400 mt-1">
            {dataset.file_count} files • Created: {dataset.created_at}
          </div>
        </div>
        <button
          onClick={() => setUploadTargetDataset(isExpanded ? null : dataset.dataset_id)}
          className="px-3 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200"
        >
          {isExpanded ? 'Close' : 'Upload'}
        </button>
      </div>

      {isExpanded && (
        <div className="border-t pt-3">
          <h4 className="text-sm font-medium mb-2">Upload Training Data</h4>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <div>
              <label className="text-xs text-gray-500">Audio File</label>
              <input
                type="file"
                accept="audio/*"
                onChange={(e) => setUploadAudioFile(e.target.files?.[0] || null)}
                className="w-full text-sm"
              />
            </div>
            <div>
              <label className="text-xs text-gray-500">Beats File (.beats)</label>
              <input
                type="file"
                accept=".beats,.txt"
                onChange={(e) => setUploadBeatsFile(e.target.files?.[0] || null)}
                className="w-full text-sm"
              />
            </div>
          </div>
          <button
            onClick={() => uploadAudioFile && uploadBeatsFile && onUpload(uploadAudioFile, uploadBeatsFile)}
            disabled={!uploadAudioFile || !uploadBeatsFile || isUploading}
            className="w-full px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            {isUploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      )}

      {dataset.audio_files.length > 0 && (
        <div className="mt-3 border-t pt-3">
          <h4 className="text-sm font-medium mb-2">Files ({dataset.file_count})</h4>
          <ul className="text-sm text-gray-600 space-y-1">
            {dataset.audio_files.map((file) => (
              <li key={file} className="flex items-center gap-2">
                <span className="text-green-500">✓</span> {file}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function TrainingJobCard({ job }: { job: TrainingJob }) {
  const statusColors = {
    queued: 'bg-yellow-100 text-yellow-800',
    running: 'bg-blue-100 text-blue-800',
    completed: 'bg-green-100 text-green-800',
    error: 'bg-red-100 text-red-800',
  };

  return (
    <div className="border rounded-lg p-4">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-medium">{job.job_name}</h3>
        <span className={`px-2 py-1 rounded text-xs ${statusColors[job.status]}`}>
          {job.status}
        </span>
      </div>

      {job.status === 'running' && (
        <div className="mb-3">
          <div className="flex justify-between text-sm mb-1">
            <span>Progress</span>
            <span>{job.progress.toFixed(1)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all"
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <div className="text-xs text-gray-500 mt-1">
            Epoch {job.epoch}/{job.total_epochs}
            {job.loss !== undefined && ` • Loss: ${job.loss.toFixed(4)}`}
          </div>
        </div>
      )}

      <div className="text-sm text-gray-600">
        <div>Dataset: {job.dataset_id}</div>
        <div>Base Model: {job.model_id}</div>
        {job.started_at && <div>Started: {job.started_at}</div>}
        {job.completed_at && <div>Completed: {job.completed_at}</div>}
        {job.error && <div className="text-red-600">Error: {job.error}</div>}
        {job.status === 'completed' && job.model_id_result && (
          <div className="text-green-600 font-medium mt-2">
            Model created: {job.model_id_result}
          </div>
        )}
      </div>
    </div>
  );
}
