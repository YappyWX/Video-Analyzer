import { useEffect, useState } from 'react';
import axios from 'axios'
import ReactMarkdown from "react-markdown";

const TreeNode = ({ node }) => {
  const [expanded, setExpanded] = useState(true);

  const hasChildren = node.children && node.children.length > 0;

  return (
    <li className="ml-4">
      <div
        className="flex items-center gap-2 cursor-pointer hover:text-green-600"
        onClick={() => hasChildren && setExpanded(!expanded)}
      >
        {hasChildren && (
          <span className="text-sm">
            {expanded ? "▼" : "▶"}
          </span>
        )}
        <span>{node.title}</span>
      </div>

      {hasChildren && expanded && (
        <ul className="list-disc ml-6">
          {node.children.map((child, index) => (
            <TreeNode key={index} node={child} />
          ))}
        </ul>
      )}
    </li>
  );
};

// Wrapper to render the full tree
const Mindmap = ({ data }) => {
  return (
    <div className="p-6 shadow-md">
      <h1 className="text-4xl font-bold mb-4 text-center">🧠 MindMap 🧠</h1>
      <h2 className="text-xl font-bold mb-4">{data.title}</h2>
      <ul>
        {data.children.map((child, index) => (
          <TreeNode key={index} node={child} />
        ))}
      </ul>
    </div>
  );
};


function App() {
  
  const [selectedFile, setSelectedFile] = useState(null)
  const [note, setNote] = useState(null)
  const [summary, setSummary] = useState(null)
  const [imgStatus, setImgStatus] = useState(false)
  const [imageNames, setImageNames] = useState([])

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0])
  }

  const handleUpload = async () => {
    if (!selectedFile) {
      alert('Please select a file')
      return;
    }

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const res = await axios.post('http://localhost:8080/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          const percentDone = Math.round((progressEvent.loaded * 100) / progressEvent.total)
          console.log(`Upload Progress: ${percentDone}`)
        }
      })
      console.log(`File uploaded, ${res.data.filename}`)
      console.log(`Processing Video...`)

      let notes = await axios.get(`http://127.0.0.1:8080/summarize/${res.data.filename}`)
      setNote(JSON.parse(notes.data.notes))
      setSummary(notes.data.summary)
      console.log(`Video Summarized!`)

      console.log("Extracting Images...")
      let imgStatus = await axios.get(`http://127.0.0.1:8080/extract/${res.data.filename}`)

      let filenames = await axios.get(`http://127.0.0.1:8080/images`)

      setImageNames(filenames.data)
      console.log(imageNames)

      if (imgStatus) {
        setImgStatus(true)
      }
      console.log("Images Extracted!")

      setSelectedFile(null)
    } catch (error) {
      console.error(error)
    }
  }

  useEffect(() => {
    console.log(summary)
  }, [note, summary, imageNames])

  return (
    <>
      <div className='w-full h-screen bg-primary'>
        <div className='w-full bg-primary grid grid-cols-6 grid-rows-3 justify-center  gap-4'>
          {/* Input Interface [ TOP ] */}
          <div className='col-span-4 h-48 col-start-2 flex justify-center items-center'>
            <input
              type="file"
              accept="video/*" // Restrict to video files
              className="text-sm text-gray-400 truncate bg-secondary rounded-l-full w-[30rem] file:truncate
                hover:cursor-pointer
                file:mr-3 file:py-2 file:px-6
                file:rounded-l-full file:border-0
                file:text-sm file:font-medium
                file:bg-main file:text-gray-200
                hover:file:cursor-pointer hover:file:bg-extra"
                onChange={handleFileChange}
            />
            <button className='bg-green-600 text-gray-200 font-medium text-sm py-2 px-6 rounded-r-full hover:cursor-pointer hover:bg-green-700' onClick={handleUpload}>Upload</button>
          </div>

          {/* Mindmap Section [ MIDDLE LEFT ] */}
          <div className='col-span-3 bg-secondary rounded-4xl border-tertiary border-4'>
            {note && (
                  <Mindmap data={note}></Mindmap>
                )}
          </div>

          {/* Summary Section [ MIDDLE RIGHT ] */}
          <div className='col-span-3 p-5 bg-secondary rounded-4xl border-tertiary border-4'>
            {summary && (
              <>
              <h1 className="text-4xl font-bold mb-4 text-center">🧠 Summary 🧠</h1>
              <ReactMarkdown>{summary}</ReactMarkdown>
              </>
            )}
          </div>

          {/* Images Section [ BOTTOM LEFT ] */}
          <div className='col-span-3 bg-secondary rounded-4xl border-tertiary border-4 grid grid-cols-5'>
            {imageNames && (
                  imageNames.map((src, idx) => (
                    <div key={idx} className='w-full h-40 overflow-hidden rounded-lg shadow'>
                        <img
                          src={`${src}`}
                          alt={`img-${idx}`}
                          className="object-cover"
                        />
                    </div>
                  ))
                )}
          </div>
        </div>
      </div>
    </>
  )
}

export default App;