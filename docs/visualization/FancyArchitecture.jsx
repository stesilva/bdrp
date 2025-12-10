import React from 'react';
import { Database, Brain, Target, BarChart3 } from 'lucide-react';

export default function FancyArchitecture() {
  const datasets = [
    { 
      name: 'CN15K', 
      description: 'Common-sense knowledge',
      nodes: '15K entities',
      edges: '205K triples'
    },
    { 
      name: 'NL27K', 
      description: 'Web-extracted facts',
      nodes: '27K entities',
      edges: '149K triples'
    },
    { 
      name: 'PPI5K', 
      description: 'Protein interactions',
      nodes: '5K entities',
      edges: '231K triples'
    }
  ];

  const methods = [
    'Normalized Edge Weights',
    'Raw Edge Weights',
    'Concatenated Edge Weights',
    'Learnable Edge Weights',
    'Probabilistic Edge Weights'
  ];

  const baselines = ['RGCN'];

  return (
    <div className="min-h-screen bg-slate-50 p-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-12">
          <div className="flex gap-2 mb-4">
            <div className="h-1 w-16 bg-slate-600 rounded"></div>
            <div className="h-1 w-16 bg-slate-400 rounded"></div>
          </div>
          <h1 className="text-4xl font-bold text-slate-900 mb-2 tracking-tight">
            Project Architecture
          </h1>
          <p className="text-slate-600 text-lg">RGCN Edge Weighting Research Pipeline</p>
        </div>

        {/* Horizontal Flow Architecture */}
        <div className="flex items-center gap-6">
          {/* Input Layer: Datasets */}
          <div className="flex-shrink-0 w-56">
            <div className="bg-white rounded-lg p-5 shadow-md border-2 border-slate-300 h-full">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-slate-700 rounded">
                  <Database className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-base font-semibold text-slate-900">Datasets</h2>
              </div>
              <div className="space-y-3">
                {datasets.map((item, idx) => (
                  <div key={idx} className="bg-slate-50 rounded p-3 border border-slate-200">
                    <div className="font-semibold text-slate-900 text-sm mb-1">{item.name}</div>
                    <div className="text-xs text-slate-600 mb-2">{item.description}</div>
                    <div className="flex gap-1 text-xs">
                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200">
                        {item.nodes}
                      </span>
                      <span className="bg-white px-2 py-0.5 rounded border border-slate-200">
                        {item.edges}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Arrow Right */}
          <div className="flex items-center flex-shrink-0">
            <div className="flex items-center">
              <div className="h-0.5 w-8 bg-slate-400"></div>
              <div className="w-3 h-3 border-t-2 border-r-2 border-slate-400 transform rotate-45"></div>
            </div>
          </div>

          {/* Processing Layer: RGCN Methods */}
          <div className="flex-shrink-0 w-64">
            <div className="bg-white rounded-lg p-5 shadow-md border-2 border-slate-300 h-full">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-slate-700 rounded">
                  <Brain className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-base font-semibold text-slate-900">RGCN Variants</h2>
              </div>
              <div className="space-y-2">
                {methods.map((method, idx) => (
                  <div key={idx} className="bg-slate-50 rounded px-3 py-2 border border-slate-200">
                    <div className="text-xs font-medium text-slate-800">
                      {method}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Arrow Right */}
          <div className="flex items-center flex-shrink-0">
            <div className="flex items-center">
              <div className="h-0.5 w-8 bg-slate-400"></div>
              <div className="w-3 h-3 border-t-2 border-r-2 border-slate-400 transform rotate-45"></div>
            </div>
          </div>

          {/* Evaluation Layer */}
          <div className="flex-shrink-0 w-56">
            <div className="bg-white rounded-lg p-5 shadow-md border-2 border-slate-300 h-full">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-slate-700 rounded">
                  <Target className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-base font-semibold text-slate-900">Link Prediction</h2>
              </div>
              <div className="space-y-3">
                <div className="bg-slate-50 rounded p-3 border border-slate-200">
                  <div className="font-semibold text-slate-900 text-sm">MR</div>
                  <div className="text-xs text-slate-600">Mean Rank</div>
                </div>
                <div className="bg-slate-50 rounded p-3 border border-slate-200">
                  <div className="font-semibold text-slate-900 text-sm">MRR</div>
                  <div className="text-xs text-slate-600">Mean Reciprocal Rank</div>
                </div>
                <div className="bg-slate-50 rounded p-3 border border-slate-200">
                  <div className="font-semibold text-slate-900 text-sm">Hits@1, 3, 10</div>
                  <div className="text-xs text-slate-600">Top-k Accuracy</div>
                </div>
              </div>
            </div>
          </div>

          {/* Arrow Right */}
          <div className="flex items-center flex-shrink-0">
            <div className="flex items-center">
              <div className="h-0.5 w-8 bg-slate-400"></div>
              <div className="w-3 h-3 border-t-2 border-r-2 border-slate-400 transform rotate-45"></div>
            </div>
          </div>

          {/* Comparison Layer: Baselines */}
          <div className="flex-shrink-0 w-48">
            <div className="bg-white rounded-lg p-5 shadow-md border-2 border-slate-300 h-full">
              <div className="flex items-center gap-2 mb-4">
                <div className="p-2 bg-slate-700 rounded">
                  <BarChart3 className="w-4 h-4 text-white" />
                </div>
                <h2 className="text-base font-semibold text-slate-900">Baseline</h2>
              </div>
              <div className="space-y-3">
                {baselines.map((baseline, idx) => (
                  <div key={idx} className="bg-slate-50 rounded p-4 border border-slate-200">
                    <div className="font-semibold text-slate-900 text-lg text-center">
                      {baseline}
                    </div>
                    <div className="text-xs text-slate-600 text-center mt-1">
                      Standard Implementation
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

