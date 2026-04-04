import { ChangeEvent, useState } from "react";

type DatasetUploadProps = {
  onUpload: (file: File) => Promise<void>;
  uploading: boolean;
};

export default function DatasetUpload({
  onUpload,
  uploading
}: DatasetUploadProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    setSelectedFile(event.target.files?.[0] ?? null);
  };

  return (
    <div className="control-stack">
      <label className="field-label">
        Upload Demand Dataset
        <input
          className="file-input"
          type="file"
          accept=".csv"
          onChange={handleFileChange}
        />
      </label>
      <button
        className="action-button"
        disabled={!selectedFile || uploading}
        onClick={() => selectedFile && onUpload(selectedFile)}
        type="button"
      >
        {uploading ? "Uploading..." : "Upload & Retrain"}
      </button>
    </div>
  );
}

