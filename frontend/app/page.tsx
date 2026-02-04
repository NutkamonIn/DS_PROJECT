'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';
import Link from 'next/link';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, Cell, LabelList 
} from 'recharts';

interface Prediction {
  model: string;
  scores: number[];
  top_class: string;
  confidence: number;
  latency_ms: number;
}

interface Benchmark {
  name: string;
  accuracy: number;
  f1: number;
  time: number;
}

interface ModelInfo {
  name: string;
  type: string;
  version: string;
}

interface ApiResponse {
  predictions: Prediction[];
  benchmarks: Benchmark[];
}

export default function Home() {
  const [text, setText] = useState<string>('');
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [modelInfos, setModelInfos] = useState<ModelInfo[]>([]);

  const exampleInputs = [
    "ตลาดหุ้นไทยปิดบวก 10 จุด รับแรงซื้อหุ้นกลุ่มพลังงานและธนาคาร",
    "นักวิทยาศาสตร์ค้นพบดาวเคราะห์ดวงใหม่ที่มีสภาพเอื้อต่อสิ่งมีชีวิต",
    "การประชุมสุดยอดผู้นำอาเซียนหารือเรื่องความร่วมมือทางเศรษฐกิจ",
    "เปิดตัวสมาร์ทโฟนรุ่นใหม่ล่าสุด กล้องชัด แบตอึด ราคาประหยัด"
  ];

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const res = await axios.get('http://localhost:8000/model/info');
        setModelInfos(res.data.models);
      } catch (err) {
        console.error("Failed to fetch model info", err);
      }
    };
    fetchModelInfo();
  }, []);

  const handlePredict = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await axios.post('http://localhost:8000/compare', { text });
      // บังคับให้ Re-render ด้วยข้อมูลใหม่
      setData(res.data);
    } catch (err) {
      console.error(err);
      alert("Cannot connect to Backend!");
    } finally {
      setLoading(false);
    }
  };

  const getModelColor = (modelName: string) => {
    if (modelName.includes("Wangchan")) return "#6366f1";
    if (modelName.includes("XLM")) return "#d946ef";
    if (modelName.includes("Logistic")) return "#14b8a6";
    if (modelName.includes("Random")) return "#f59e0b";
    return "#64748b";
  };

  return (
    <div className="min-h-screen bg-slate-50 p-8 font-sans text-slate-900">
      
      {/* Header */}
      <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center mb-10 gap-4">
        <div className="text-center md:text-left">
          <h1 className="text-5xl font-extrabold text-indigo-900 mb-3 tracking-tight">AI Model Arena</h1>
          <p className="text-slate-500 text-lg">
            Comparing Thai News Classification Models: <span className="font-semibold text-indigo-600">Deep Learning</span> vs <span className="font-semibold text-teal-600">Classic ML</span>
          </p>
        </div>
        <div className="flex gap-3">
          <Link href="/comparison" className="px-5 py-3 bg-slate-100 text-slate-700 rounded-xl font-bold hover:bg-slate-200 transition-all shadow-sm">
            Model Comparison
          </Link>
          <Link href="/evaluation" className="px-6 py-3 bg-indigo-100 text-indigo-700 rounded-xl font-bold hover:bg-indigo-200 transition-all shadow-sm">
            Evaluation Dashboard
          </Link>
        </div>
      </div>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left: Input */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <label className="block text-slate-700 font-bold mb-3">Input News Content</label>
            <textarea 
              value={text} 
              onChange={(e) => setText(e.target.value)} 
              placeholder="Paste Thai news here..." 
              rows={6}
              className="w-full p-4 border border-slate-200 rounded-xl outline-none focus:border-indigo-500 transition-all resize-none"
            />
            <button 
              onClick={handlePredict} 
              disabled={loading}
              className={`w-full mt-4 p-3 rounded-xl text-white font-bold text-lg transition-all ${loading ? 'bg-slate-400' : 'bg-indigo-600 hover:bg-indigo-700'}`}
            >
              {loading ? 'Processing...' : 'Predict / Analyze'}
            </button>
          </div>

          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
            <h3 className="font-bold mb-3 text-sm uppercase">Try Examples</h3>
            <div className="flex flex-col gap-2">
              {exampleInputs.map((ex, i) => (
                <button key={i} onClick={() => setText(ex)} className="text-left text-sm text-slate-600 p-2 rounded hover:bg-slate-50 border border-transparent hover:border-slate-200 truncate">
                  {ex}
                </button>
              ))}
            </div>
          </div>

          {modelInfos.length > 0 && (
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-100">
              <h3 className="font-bold mb-3 text-sm uppercase">Model Info</h3>
              <div className="space-y-3">
                {modelInfos.map((info, i) => (
                  <div key={i} className="text-xs border-b border-slate-50 pb-2 last:border-0">
                    <div className="font-semibold">{info.name}</div>
                    <div className="text-slate-400">{info.type}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-8 flex flex-col gap-6">
          {data ? (
            <>
              {/* Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {data.predictions.map((model, index) => (
                  <div key={index} className="bg-white p-5 rounded-2xl shadow-sm border-t-4" style={{borderColor: getModelColor(model.model)}}>
                    <div className="flex justify-between items-start mb-4">
                      <div>
                        <h3 className="text-slate-400 font-bold text-[10px] uppercase">{model.model}</h3>
                        <h2 className="text-xl font-black text-slate-800">{model.top_class}</h2>
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-green-50 text-green-600">
                          {(model.confidence * 100).toFixed(1)}% Confidence
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-[10px] text-slate-400">Latency</div>
                        <div className="font-bold text-indigo-600 text-sm">{model.latency_ms} ms</div>
                      </div>
                    </div>
                    
                    <div className="h-20">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={[
                          { name: 'World', val: model.scores[0] * 100 },
                          { name: 'Business', val: model.scores[1] * 100 },
                          { name: 'SciTech', val: model.scores[2] * 100 },
                        ]}>
                          <XAxis dataKey="name" tick={{fontSize: 10}} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{borderRadius: '8px', fontSize: '10px'}} // eslint-disable-next-line @typescript-eslint/no-explicit-any
                          formatter={(val: any) => [`${Number(val).toFixed(1)}%`]} />
                          <Bar dataKey="val" radius={[4, 4, 0, 0]}>
                            {[0,1,2].map((_, i) => <Cell key={i} fill={getModelColor(model.model)} />)}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                ))}
              </div>

              {/* Benchmark Charts */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                  <h3 className="text-slate-700 font-bold mb-4 text-sm uppercase">Accuracy & F1</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.benchmarks} margin={{ bottom: 40 }}>
                        <XAxis dataKey="name" tick={{fontSize: 9}} interval={0} angle={-25} textAnchor="end" />
                        <YAxis domain={[0, 100]} tick={{fontSize: 10}} />
                        <Tooltip contentStyle={{borderRadius: '8px', fontSize: '12px'}} />
                        <Legend wrapperStyle={{fontSize: '10px'}} />
                        <Bar dataKey="accuracy" fill="#10b981" name="Acc" />
                        <Bar dataKey="f1" fill="#3b82f6" name="F1" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-100">
                  <h3 className="text-slate-700 font-bold mb-4 text-sm uppercase">Live Latency (ms)</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.benchmarks} layout="vertical" margin={{ left: 30, right: 30 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="name" type="category" width={80} tick={{fontSize: 10}} />
                        <Tooltip // eslint-disable-next-line @typescript-eslint/no-explicit-any
                        formatter={(val: any) => `${val} ms`} />
                        <Bar dataKey="time" fill="#f59e0b" name="Latency">
                          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                          <LabelList dataKey="time" position="right" style={{fontSize: '10px'}} />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="h-full flex items-center justify-center bg-white rounded-2xl border-2 border-dashed border-slate-200 p-10 min-h-[400px]">
              <p className="text-slate-400">Enter news content on the left to start AI analysis</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}