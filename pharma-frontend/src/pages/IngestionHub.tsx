import React, { useState } from 'react';
import axios from 'axios';

const IngestionHub = () => {
  const [file, setFile] = useState<File | null>(null);
  const [comment, setComment] = useState('');
  const [uploader, setUploader] = useState('');
  const [table, setTable] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError("Please select a file first");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("uploader_name", uploader);
      formData.append("table_name", table);
      formData.append("comment", comment);
      formData.append("file", file);

      const response = await axios.post(
        "http://localhost:8000/api/injection/upload",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      setResult(response.data);
    } catch (err: any) {
      console.error("Upload Error:", err);
      setError(
        err.response?.data?.detail ||
          "Error uploading file. Check backend connection or file format."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-slate-50 p-10">
      {/* Header */}
      <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-lg p-8 border border-slate-200">
        <h1 className="text-3xl font-bold text-slate-800 mb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
          Ingestion Hub
        </h1>
        <p className="text-slate-500 mb-8">Upload and manage PDF, DOCX, or CSV files.</p>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Uploader Name */}
          <div>
            <label className="block text-slate-700 font-semibold mb-2">Uploader Name</label>
            <input
              type="text"
              value={uploader}
              onChange={(e) => setUploader(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-primary focus:outline-none"
              placeholder="Enter your name"
              required
            />
          </div>

          {/* Table Name */}
          <div>
            <label className="block text-slate-700 font-semibold mb-2">Table Name</label>
            <input
              type="text"
              value={table}
              onChange={(e) => setTable(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-primary focus:outline-none"
              placeholder="e.g., Target_list"
              required
            />
          </div>

          {/* Comment */}
          <div>
            <label className="block text-slate-700 font-semibold mb-2">Comment</label>
            <input
              type="text"
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              className="w-full px-4 py-2 border border-slate-300 rounded-xl focus:ring-2 focus:ring-primary focus:outline-none"
              placeholder="Describe your upload (e.g., 'Remove Dr. Ishika')"
              required
            />
          </div>

          {/* File Upload */}
          <div>
            <label className="block text-slate-700 font-semibold mb-2">File</label>
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="w-full text-slate-600 border border-slate-300 rounded-xl p-2 focus:ring-2 focus:ring-primary focus:outline-none"
              required
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="px-6 py-3 bg-gradient-to-br from-primary to-secondary text-white rounded-xl font-semibold hover:shadow-lg transition-all duration-300 hover:scale-105 disabled:opacity-60"
          >
            {loading ? "Uploading..." : "Upload File"}
          </button>
        </form>

        {/* Feedback Section */}
        {error && (
          <p className="mt-6 text-red-500 font-semibold">{error}</p>
        )}

        {/* {result && (
          <div className="mt-8 border-t pt-4">
            <h2 className="text-lg font-semibold text-green-700">✅ Upload Successful</h2>
            <pre className="mt-2 bg-gray-100 p-3 rounded-lg text-sm text-gray-800 overflow-auto">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )} */}
        {result && (
  <p className="mt-6 text-green-600 font-semibold text-lg flex items-center gap-2">
    ✅ uploaded successfully!
  </p>
)}
      
      </div>
    </div>
  );
};

export default IngestionHub;
