import React, { useState } from 'react';
import { Upload, CheckCircle, AlertCircle, FileText, Loader } from 'lucide-react';

const IngestionHub = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploader, setUploader] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [dragActive, setDragActive] = useState(false);

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    if (!uploader) {
      setError("Please enter your name");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("uploader_name", uploader);
      formData.append("file", file);

      const response = await fetch(
        "http://localhost:8000/api/injection/upload",
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Upload failed");
      }

      const data = await response.json();
      setResult(data);
      setFile(null);
      setUploader('');
    } catch (err: any) {
      console.error("Upload Error:", err);
      setError(err.message || "Error uploading file. Check backend connection.");
    } finally {
      setLoading(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-cyan-50 p-6 flex items-center justify-center">
      <div className="w-full max-w-2xl">
        {/* Main Card */}
        <div className="bg-white rounded-3xl shadow-2xl p-8 border border-gray-200">
          {/* Header */}
          <div className="text-center mb-10">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-2xl mb-4 shadow-lg">
              <Upload className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-4xl font-bold text-gray-900 mb-2">Ingestion Hub</h1>
            <p className="text-gray-600">Upload PDF, DOCX, or TXT files with ease</p>
          </div>

          <div className="space-y-6">
            {/* Uploader Name */}
            <div>
              <label className="block text-gray-700 font-semibold mb-3">Uploader Name</label>
              <input
                type="text"
                value={uploader}
                onChange={(e) => setUploader(e.target.value)}
                className="w-full px-4 py-3 bg-gray-50 border border-gray-300 rounded-xl text-gray-900 placeholder-gray-400 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/20 focus:outline-none transition-all"
                placeholder="Enter your name"
              />
            </div>

            {/* File Upload - Drag & Drop */}
            <div
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              className={`relative border-2 border-dashed rounded-2xl p-8 transition-all duration-300 cursor-pointer ${
                dragActive
                  ? 'border-cyan-500 bg-cyan-50'
                  : 'border-gray-300 bg-gray-50 hover:border-gray-400 hover:bg-gray-100'
              }`}
            >
              <input
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
              
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-14 h-14 bg-gradient-to-br from-cyan-100 to-blue-100 rounded-xl mb-4">
                  <FileText className="w-7 h-7 text-cyan-600" />
                </div>
                <p className="text-gray-900 font-semibold mb-1">
                  {file ? file.name : 'Drop your file here or click to browse'}
                </p>
                <p className="text-gray-500 text-sm">
                  Supported: PDF, DOCX, TXT
                </p>
              </div>
            </div>

            {/* Submit Button */}
            <button
              onClick={handleSubmit}
              disabled={loading || !file || !uploader}
              className="w-full py-3 bg-gradient-to-r from-blue-500 to-cyan-500 text-white font-semibold rounded-xl hover:shadow-lg hover:shadow-cyan-500/50 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  Upload File
                </>
              )}
            </button>
          </div>

          {/* Success Message */}
          {result && (
            <div className="mt-8 p-4 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-xl">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-6 h-6 text-green-600 flex-shrink-0" />
                <div>
                  <p className="text-green-700 font-semibold">Successfully uploaded!</p>
                  <p className="text-green-600 text-sm mt-1">Your file has been processed.</p>
                </div>
              </div>
            </div>
          )}

          {/* Error Message */}
          {error && (
            <div className="mt-8 p-4 bg-gradient-to-r from-red-50 to-pink-50 border border-red-200 rounded-xl">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0" />
                <div>
                  <p className="text-red-700 font-semibold">Upload failed</p>
                  <p className="text-red-600 text-sm mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer Info */}
        <div className="mt-8 text-center text-gray-500 text-sm">
          <p>Your files are processed securely and stored safely</p>
        </div>
      </div>
    </div>
  );
};

export default IngestionHub;