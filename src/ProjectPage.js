import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

const BaselineSection = () => {
    const baselineDetails = {
      traditional: [
        {
          name: "Random Baseline",
          description: "Randomly assigns stance labels",
          performance: { f1: 0.335, accuracy: 0.335 },
          implementation: `
            class RandomBaseline:
                def __init__(self, n_classes=3):
                    self.n_classes = n_classes
                    
                def predict(self, loader):
                    return random.randint(0, self.n_classes-1)
          `,
          details: "Serves as a lower bound for model performance evaluation"
        },
        {
          name: "Majority Class",
          description: "Always predicts the most frequent stance label",
          performance: { f1: 0.169, accuracy: 0.169 },
          implementation: `
            class MajorityClassBaseline:
                def __init__(self):
                    self.majority_class = None
                    
                def fit(self, loader):
                    # Find most common class
                    self.majority_class = np.argmax(np.bincount(all_labels))
                
                def predict(self, loader):
                    return self.majority_class
          `,
          details: "Provides baseline for imbalanced dataset scenarios"
        }
      ],
      ml_based: [
        {
          name: "TF-IDF + Naive Bayes",
          description: "TF-IDF features with Multinomial Naive Bayes classifier",
          performance: { f1: 0.343, accuracy: 0.343 },
          features: [
            "TF-IDF vectorization of text",
            "Separate vectorization for topics",
            "Multinomial Naive Bayes classification",
            "Feature concatenation strategy"
          ],
          technical_details: "Uses 10,000 max features for text and 5,000 for topics"
        },
        {
          name: "TF-IDF + Random Forest",
          description: "TF-IDF features with Random Forest classifier",
          performance: { f1: 0.297, accuracy: 0.297 },
          features: [
            "TF-IDF text representation",
            "Ensemble of decision trees",
            "Feature importance analysis",
            "Non-linear decision boundaries"
          ],
          technical_details: "100 trees, balanced class weights"
        },
        {
          name: "TF-IDF + Logistic Regression",
          description: "TF-IDF features with logistic regression",
          performance: { f1: 0.331, accuracy: 0.331 },
          features: [
            "Linear classification model",
            "L2 regularization",
            "Probability calibration",
            "Multi-class strategy: one-vs-rest"
          ],
          technical_details: "C=1.0, max_iter=1000, 'liblinear' solver"
        }
      ],
      llm_based: [
        {
          name: "GPT-4 with Structured Output",
          description: "Large language model with constrained output format",
          performance: { f1: 0.21, accuracy: 0.20 },
          setup: {
            model: "gpt-4o-2024-08-06",
            output_format: {
              stance: "agree/disagree/neutral",
              confidence: "float(0-1)",
              explanation: "Optional[str]"
            },
            prompt_structure: "Zero-shot prompting with explicit stance instructions"
          },
          limitations: [
            "Constrained by structured output format",
            "Limited vocabulary in stance expressions",
            "No task-specific training",
            "Fixed response format restrictions"
          ],
          implementation_details: `
            response = client.chat.completions.parse(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Text: {text}\\nTopic: {topic}"}
                ],
                response_format=StanceResponse
            )
          `
        }
      ]
    };
  
    return (
      <div className="space-y-8">
        <h2 className="text-2xl font-bold mb-6">Baseline Approaches</h2>
  
        {/* Traditional Baselines */}
        <section className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-bold mb-4">Traditional Baselines</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {baselineDetails.traditional.map((baseline, idx) => (
              <div key={idx} className="border rounded-lg p-4">
                <h4 className="font-bold text-lg mb-2">{baseline.name}</h4>
                <p className="text-gray-600 mb-3">{baseline.description}</p>
                <div className="bg-gray-50 p-3 rounded mb-3">
                  <div className="text-sm font-mono">{baseline.implementation}</div>
                </div>
                <div className="flex justify-between text-sm">
                  <span>F1: {baseline.performance.f1}</span>
                  <span>Accuracy: {baseline.performance.accuracy}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
  
        {/* ML-Based Approaches */}
        <section className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-bold mb-4">Machine Learning Baselines</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {baselineDetails.ml_based.map((baseline, idx) => (
              <div key={idx} className="border rounded-lg p-4">
                <h4 className="font-bold text-lg mb-2">{baseline.name}</h4>
                <p className="text-gray-600 mb-3">{baseline.description}</p>
                <div className="space-y-2 mb-3">
                  {baseline.features.map((feature, fidx) => (
                    <div key={fidx} className="flex items-start">
                      <span className="text-blue-500 mr-2">•</span>
                      <span className="text-sm">{feature}</span>
                    </div>
                  ))}
                </div>
                <div className="bg-gray-50 p-3 rounded text-sm">
                  <p>{baseline.technical_details}</p>
                </div>
                <div className="flex justify-between text-sm mt-3">
                  <span>F1: {baseline.performance.f1}</span>
                  <span>Accuracy: {baseline.performance.accuracy}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
  
        {/* LLM-Based Approach */}
        <section className="bg-white rounded-lg shadow p-6">
          <h3 className="text-xl font-bold mb-4">LLM-Based Baseline</h3>
          <div className="space-y-4">
            {baselineDetails.llm_based.map((baseline, idx) => (
              <div key={idx} className="border rounded-lg p-6">
                <h4 className="font-bold text-lg mb-2">{baseline.name}</h4>
                <p className="text-gray-600 mb-4">{baseline.description}</p>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h5 className="font-semibold mb-2">Setup</h5>
                    <div className="bg-gray-50 p-3 rounded">
                      <p className="text-sm mb-2">Model: {baseline.setup.model}</p>
                      <p className="text-sm mb-2">Output Format:</p>
                      <pre className="text-xs">
                        {JSON.stringify(baseline.setup.output_format, null, 2)}
                      </pre>
                    </div>
                  </div>
                  
                  <div>
                    <h5 className="font-semibold mb-2">Limitations</h5>
                    <ul className="space-y-1">
                      {baseline.limitations.map((limitation, lidx) => (
                        <li key={lidx} className="flex items-start">
                          <span className="text-red-500 mr-2">•</span>
                          <span className="text-sm">{limitation}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
  
                <div className="mt-4">
                  <h5 className="font-semibold mb-2">Implementation</h5>
                  <div className="bg-gray-50 p-3 rounded">
                    <pre className="text-xs overflow-x-auto">
                      {baseline.implementation_details}
                    </pre>
                  </div>
                </div>
  
                <div className="flex justify-between text-sm mt-4 bg-gray-100 p-3 rounded">
                  <span>F1 Score: {baseline.performance.f1}</span>
                  <span>Accuracy: {baseline.performance.accuracy}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    );
  };
const ProjectPage = () => {
  // State for expandable sections
  const [expandedSection, setExpandedSection] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [activeMethodTab, setActiveMethodTab] = useState('tganet');

  // Zero-shot performance data
  const zeroShotResults = [
    { name: 'Random', score: 0.335 },
    { name: 'Majority', score: 0.169 },
    { name: 'Naive Bayes', score: 0.343 },
    { name: 'Random Forest', score: 0.297 },
    { name: 'Log Regr.', score: 0.331 },
    { name: 'GPT-4', score: 0.21 },
    { name: 'TGANet', score: 0.666 },
    { name: 'Mod. TGANet', score: 0.354 },
    { name: 'TESTED', score: 0.594 }
  ];

  // Parameter study results
  const paramStudyResults = [
    { config: 'BS16_LR1e-4_D0.0', score: 0.324 },
    { config: 'BS16_LR1e-4_D0.2', score: 0.354 },
    { config: 'BS16_LR2e-5_D0.0', score: 0.340 },
    { config: 'BS32_LR1e-4_D0.2', score: 0.342 },
    { config: 'BS32_LR2e-5_D0.0', score: 0.335 },
    { config: 'BS32_LR2e-5_D0.2', score: 0.332 }
  ];

  // Dataset examples with full annotation details
  const datasetExamples = {
    positive: [
      {
        text: "Medical websites provide valuable health information and improve doctor-patient communication.",
        topic: "medical websites",
        annotation_type: "Heur",
        details: "Direct positive stance with reasoning"
      }
    ],
    negative: [
      {
        text: "The jury's verdict will ensure that another violent criminal alien will be removed from our community.",
        topic: "immigration",
        annotation_type: "List",
        details: "Implicit negative stance through context"
      }
    ],
    neutral: [
      {
        text: "Studies have shown both benefits and risks in implementing high-speed rail systems.",
        topic: "high-speed rail",
        annotation_type: "Corr",
        details: "Balanced presentation without clear stance"
      }
    ]
  };

  // Dataset statistics
  const datasetStats = {
    total_examples: 23525,
    unique_topics: 5634,
    few_shot: {
      train: 13477,
      dev: 2062,
      test: 3006,
      topics: {
        few_shot: {
          train: 638,
          dev: 114,
          test: 159
        },
        zero_shot: {
          train: 4003,
          dev: 383,
          test: 600
        }
      }
    },
    annotation_types: {
      heur: {
        count: 4416,
        pro_percent: 49,
        con_percent: 51,
        description: "Original topic extraction"
      },
      corr: {
        count: 3594,
        pro_percent: 44,
        con_percent: 51,
        description: "Corrected topic annotations"
      },
      list: {
        count: 11531,
        pro_percent: 50,
        con_percent: 48,
        description: "Annotator-listed topics"
      },
      neutral: {
        count: 3984,
        pro_percent: 0,
        con_percent: 0,
        description: "Neutral stance examples"
      }
    },
    unique_comments: {
      train: 1845,
      dev: 682,
      test: 786
    },
    paper_citation: "Allaway & McKeown (2020)",
    dataset_source: "The New York Times 'Room for Debate' section, part of the Argument Reasoning Comprehension (ARC) Corpus (Habernal et al., 2018)"
  };

  // Navigation links
  const navLinks = [
    ['overview', 'Overview'],
    ['dataset', 'VAST Dataset'],
    ['methods', 'Methods'],
    ['results', 'Results'],
    ['baselines', 'Baselines'],  // Add this line
    ['discussion', 'Discussion']
  ];

  // Method details for each model
  const methodDetails = {
    tganet: {
      name: "TGANet (Original)",
      description: "Topic-Grouped Attention Network for zero-shot stance detection, featuring a novel architecture for topic-aware stance classification",
      implementation_details: {
        framework: "PyTorch",
        base_model: "BERT-base-uncased",
        training: {
          optimizer: "AdamW",
          learning_rate: "2e-5",
          batch_size: 16,
          epochs: 10,
          warmup_steps: 500,
          weight_decay: 0.01
        }
      },
      architecture: [
        {
          component: "Generalized Topic Representations (GTR)",
          details: [
            "BERT embeddings for topic representation",
            "Hierarchical clustering for topic grouping",
            "Centroid-based topic representations",
            "Enables zero-shot learning through topic similarity"
          ],
          technical: `Uses BERT to generate embeddings followed by clustering with
          topic vectors obtained through joint learning of document and word semantics.
          Implements scaled dot-product attention following Vaswani et al. (2017).`
        },
        {
          component: "Topic-Grouped Attention",
          details: [
            "Multi-head attention mechanism",
            "Dynamic topic weighting",
            "Context-aware feature extraction",
            "Topic-specific attention patterns"
          ],
          technical: `Attention computed as: si = softmax(t(i) · (Wardt)),
          where Wa ∈ RE×2E are learned parameters and t(i) represents topic tokens.
          Uses a scaling factor of 1/√E for numerical stability.`
        },
        {
          component: "Classification Layer",
          details: [
            "Feed-forward neural network",
            "Dropout regularization",
            "Cross-entropy loss",
            "Softmax activation"
          ],
          technical: `Final classification computed as:
          p = softmax(W2(tanh(W1[d̃; cdt] + b1)) + b2)
          where d̃ is pooled document representation and cdt is topic-grouped attention output.`
        }
      ],
      performance_analysis: {
        overall_metrics: {
          f1_score: 0.666,
          accuracy: 0.665,
          precision: 0.664,
          recall: 0.667
        },
        strengths: [
          "Superior zero-shot generalization",
          "Effective topic relationship modeling",
          "Robust to topic variations",
          "Strong performance on implicit stances"
        ],
        limitations: [
          "Computationally intensive",
          "Requires significant training data",
          "Complex hyperparameter tuning",
          "Sensitive to topic clustering quality"
        ]
      }
    },
    tested: {
      name: "TESTED Framework",
      description: "Topic Efficient StancE Detection with balanced sampling and contrastive learning",
      implementation_details: {
        framework: "PyTorch",
        base_model: "BERT-base-uncased",
        training: {
          optimizer: "AdamW",
          learning_rate: "3e-5",
          batch_size: 32,
          epochs: 8,
          warmup_steps: 200,
          weight_decay: 0.01
        }
      },
      architecture: [
        {
          component: "Topic-Guided Sampling",
          details: [
            "Data-efficient sampling technique",
            "Diversity preservation",
            "Balance maintenance",
            "Topic representation sampling"
          ],
          technical: `Implements importance-weighted topic-guided diversity sampling
          with cluster-based selection. Uses stratified sampling to maintain class balance.`
        },
        {
          component: "Contrastive Learning",
          details: [
            "Custom contrastive objective",
            "Similarity-based alignment",
            "Enhanced feature discrimination",
            "Topic-aware contrast"
          ],
          technical: `Uses contrastive loss with exponential penalization for
          opposing pairs and similarity maximization for aligned pairs.
          Implements temperature-scaled InfoNCE loss for contrastive learning.`
        }
      ],
      performance_analysis: {
        overall_metrics: {
          f1_score: 0.594,
          accuracy: 0.592,
          precision: 0.595,
          recall: 0.593
        },
        strengths: [
          "Efficient training process",
          "Good performance with limited data",
          "Balanced class handling",
          "Effective feature learning"
        ],
        limitations: [
          "Lower performance than TGANet",
          "Sensitive to sampling strategy",
          "Complex training dynamics",
          "Limited zero-shot capability"
        ]
      }
    },
    modified_tganet: {
      name: "Modified TGANet",
      description: "Our modified version with enhanced features and architectural improvements",
      implementation_details: {
        framework: "PyTorch",
        base_model: "BERT-base-uncased",
        training: {
          optimizer: "AdamW",
          learning_rate: "2e-5",
          batch_size: 16,
          epochs: 15,
          warmup_steps: 1000,
          weight_decay: 0.02
        }
      },
      architecture: [
        {
          component: "Enhanced Topic Clustering",
          details: [
            "Hierarchical agglomerative clustering",
            "Optimized linkage criteria",
            "Dynamic cluster size adjustment",
            "Improved topic grouping"
          ],
          technical: `Modified Ward's method with optimized cluster selection and
          dynamic size adjustment based on topic distribution. Implements adaptive
          threshold for cluster formation.`
        },
        {
          component: "Additional Attention Mechanisms",
          details: [
            "Increased attention heads",
            "Enhanced topic-text relationships",
            "Modified attention computation",
            "Improved feature capture"
          ],
          technical: `Extended attention mechanism with additional heads and
          modified computation for better topic-text relationship modeling.
          Implements multi-scale attention with varying window sizes.`
        }
      ],
      modifications: [
        {
          change: "Topic Clustering",
          reason: "Improve topic grouping stability",
          impact: "Mixed results with some loss in performance",
          details: "Modified clustering algorithm showed more consistent grouping but slightly reduced overall performance"
        },
        {
          change: "Attention Mechanism",
          reason: "Better feature capture",
          impact: "Increased complexity without proportional gains",
          details: "Additional attention heads increased model complexity but provided marginal improvements"
        },
        {
          change: "Training Process",
          reason: "Enhanced stability",
          impact: "Slower convergence with similar final results",
          details: "Modified learning rate schedule and gradient accumulation for stability"
        }
      ],
      performance_analysis: {
        overall_metrics: {
          f1_score: 0.354,
          accuracy: 0.352,
          precision: 0.355,
          recall: 0.353
        },
        strengths: [
          "More stable training",
          "Better feature interpretability",
          "Improved topic clustering",
          "Enhanced attention patterns"
        ],
        limitations: [
          "Lower performance than original",
          "Increased computational cost",
          "More complex architecture",
          "Harder to tune"
        ]
      }
    }
  };

  // Add new constants for detailed results
  const phenomenonResults = {
    imp: {
      name: "Implicit Stance",
      description: "Cases where the topic is not explicitly mentioned in the text",
      tganet: { in: 0.623, out: 0.713 },
      bert_joint: { in: 0.600, out: 0.710 }
    },
    mlT: {
      name: "Multiple Topics",
      description: "Documents containing more than one topic",
      tganet: { in: 0.624, out: 0.752 },
      bert_joint: { in: 0.610, out: 0.748 }
    },
    mlS: {
      name: "Multiple Stance",
      description: "Documents with different non-neutral stance labels",
      tganet: { in: 0.547, out: 0.725 },
      bert_joint: { in: 0.541, out: 0.713 }
    },
    qte: {
      name: "Quotations",
      description: "Documents containing quoted text",
      tganet: { in: 0.661, out: 0.663 },
      bert_joint: { in: 0.625, out: 0.657 }
    },
    sarc: {
      name: "Sarcasm",
      description: "Documents containing sarcastic content",
      tganet: { in: 0.637, out: 0.667 },
      bert_joint: { in: 0.587, out: 0.662 }
    }
  };

  const modelPerformance = {
    development: {
      metrics: {
        CMaj: { f1_all: 0.3817, f1_zero: 0.2504, f1_few: 0.2910 },
        BoWV: { f1_all: 0.3367, f1_zero: 0.3213, f1_few: 0.3493 },
        'C-FFNN': { f1_all: 0.3307, f1_zero: 0.3147, f1_few: 0.3464 },
        BiCond: { f1_all: 0.4229, f1_zero: 0.4272, f1_few: 0.4170 },
        'Cross-Net': { f1_all: 0.4779, f1_zero: 0.4601, f1_few: 0.4942 },
        'BERT-sep': { f1_all: 0.5314, f1_zero: 0.5109, f1_few: 0.5490 },
        'BERT-joint': { f1_all: 0.6589, f1_zero: 0.6375, f1_few: 0.6099 },
        'TGA Net': { f1_all: 0.6657, f1_zero: 0.6851, f1_few: 0.6421 }
      }
    }
  };

  // Add new constants for combined results
  const combinedResults = {
    paper_results: {
      'TGA Net': { accuracy: 0.665, f1: 0.666, f1_unseen: 0.666 },
      'BERT-joint': { accuracy: 0.653, f1: 0.653, f1_unseen: 0.661 },
      'BERT-sep': { accuracy: 0.501, f1: 0.501, f1_unseen: 0.454 },
      'Cross-Net': { accuracy: 0.455, f1: 0.455, f1_unseen: 0.434 },
      'BiCond': { accuracy: 0.415, f1: 0.415, f1_unseen: 0.428 },
      'C-FFNN': { accuracy: 0.300, f1: 0.300, f1_unseen: 0.417 }
    },
    our_implementation: {
      'TGANet': { accuracy: 0.665, f1: 0.666, f1_unseen: 0.666 },
      'TESTED': { accuracy: 0.592, f1: 0.594, f1_unseen: 0.594 },
      'Modified TGANet': { accuracy: 0.352, f1: 0.354, f1_unseen: 0.354 },
      'LLM (GPT-4)': { accuracy: 0.200, f1: 0.199, f1_unseen: 0.199 },
      'Naive Bayes': { accuracy: 0.417, f1: 0.343, f1_unseen: 0.343 },
      'Random Forest': { accuracy: 0.365, f1: 0.297, f1_unseen: 0.297 },
      'Logistic Regression': { accuracy: 0.400, f1: 0.331, f1_unseen: 0.331 },
      'Random': { accuracy: 0.335, f1: 0.335, f1_unseen: 0.335 },
      'Majority Class': { accuracy: 0.339, f1: 0.169, f1_unseen: 0.169 }
    }
  };

  // Add confusion matrix data
  const confusionMatrices = {
    'Naive Bayes': [
      [569, 440, 9],
      [265, 672, 7],
      [449, 582, 13]
    ],
    'Random Forest': [
      [571, 447, 0],
      [425, 519, 0],
      [527, 511, 6]
    ],
    'Logistic Regression': [
      [528, 476, 14],
      [277, 655, 12],
      [448, 578, 18]
    ]
  };

  // Add class-wise performance data
  const classPerformance = {
    'TGANet': {
      'agree': { precision: 0.573, recall: 0.585, f1: 0.579 },
      'disagree': { precision: 0.590, recall: 0.595, f1: 0.592 },
      'neutral': { precision: 0.665, recall: 0.666, f1: 0.665 }
    },
    'TESTED': {
      'agree': { precision: 0.443, recall: 0.559, f1: 0.495 },
      'disagree': { precision: 0.397, recall: 0.712, f1: 0.509 },
      'neutral': { precision: 0.448, recall: 0.012, f1: 0.024 }
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <nav className="sticky top-0 bg-white shadow-sm z-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8 py-4">
            {navLinks.map(([id, label]) => (
              <button
                key={id}
                className={`px-3 py-2 rounded-md ${
                  activeTab === id ? 'bg-blue-100 text-blue-800' : 'text-gray-600'
                }`}
                onClick={() => setActiveTab(id)}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-blue-800 text-white py-20">
        <div className="max-w-7xl mx-auto px-4">
          <h1 className="text-4xl font-bold mb-4">Zero-Shot Stance Detection</h1>
          <p className="text-xl opacity-90">
            Comprehensive Analysis of Topic-Grouped Attention Networks
          </p>
        </div>
      </div>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-12">
        {/* Content sections based on activeTab */}
        {activeTab === 'overview' && (
            <div className="space-y-12">
                <section>
                <h2 className="text-2xl font-bold mb-6">Introduction</h2>
                <div className="prose max-w-none">
                    <p className="mb-4">
                    Stance detection is a crucial task in natural language processing that aims to identify whether a text expresses 
                    support, opposition, or neutrality towards a specific topic. The challenge becomes particularly complex in zero-shot 
                    scenarios where models must generalize to previously unseen topics.
                    </p>
                    
                    <blockquote className="bg-gray-50 border-l-4 border-blue-500 p-4 my-6">
                    <p className="italic">
                        "Stance Detection is concerned with identifying the attitudes expressed by an author towards a target of 
                        interest. This task spans a variety of domains ranging from social media opinion identification to detecting 
                        the stance for a legal claim."
                    </p>
                    <footer className="mt-2 text-sm">- VAST Paper</footer>
                    </blockquote>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-8">
                    <div className="bg-white rounded-lg shadow-md p-6">
                        <h3 className="text-xl font-bold mb-4">Key Challenges</h3>
                        <ul className="space-y-2">
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            <span>Generalizing to unseen topics without prior training</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            <span>Handling implicit and nuanced stance expressions</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            <span>Managing diverse text styles across domains</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            <span>Balancing model complexity with performance</span>
                        </li>
                        </ul>
                    </div>

                    <div className="bg-white rounded-lg shadow-md p-6">
                        <h3 className="text-xl font-bold mb-4">Our Approach</h3>
                        <ul className="space-y-2">
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">✓</span>
                            <span>Implementation and evaluation of TGANet architecture</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">✓</span>
                            <span>Modified TGANet with enhanced features</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">✓</span>
                            <span>Comparison with TESTED and LLM approaches</span>
                        </li>
                        <li className="flex items-start">
                            <span className="text-green-500 mr-2">✓</span>
                            <span>Comprehensive evaluation on VAST dataset</span>
                        </li>
                        </ul>
                    </div>
                    </div>

                    <div className="bg-white rounded-lg shadow-md p-6 mt-8">
                    <h3 className="text-xl font-bold mb-4">Project Contributions</h3>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div>
                        <h4 className="font-semibold mb-2">Implementation</h4>
                        <p className="text-sm text-gray-600">
                            Comprehensive implementation and analysis of state-of-the-art stance detection models
                        </p>
                        </div>
                        <div>
                        <h4 className="font-semibold mb-2">Evaluation</h4>
                        <p className="text-sm text-gray-600">
                            Detailed comparison of different approaches including traditional ML, neural, and LLM methods
                        </p>
                        </div>
                        <div>
                        <h4 className="font-semibold mb-2">Analysis</h4>
                        <p className="text-sm text-gray-600">
                            In-depth analysis of model behaviors and performance characteristics in zero-shot scenarios
                        </p>
                        </div>
                    </div>
                    </div>
                </div>
                </section>
            </div>
        )}
        {activeTab === 'dataset' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-6">VAST Dataset Analysis</h2>
              
              {/* Dataset Overview */}
              <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h3 className="text-xl font-bold mb-4">Dataset Overview</h3>
                <p className="text-gray-700 mb-4">
                  The VAST dataset, introduced by {datasetStats.paper_citation}, is collected from {datasetStats.dataset_source}. 
                  It features a large number of specific topics, making it ideal for zero-shot stance detection research.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">Key Statistics</h4>
                    <ul className="space-y-2">
                      <li>Total Examples: {datasetStats.total_examples}</li>
                      <li>Unique Topics: {datasetStats.unique_topics}</li>
                      <li>Median examples per topic: 1</li>
                      <li>Mean examples per topic: 2.4</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Data Distribution</h4>
                    <ul className="space-y-2">
                      <li>Training Set: {datasetStats.few_shot.train} examples</li>
                      <li>Development Set: {datasetStats.few_shot.dev} examples</li>
                      <li>Test Set: {datasetStats.few_shot.test} examples</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Annotation Types */}
              <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h3 className="text-xl font-bold mb-4">Annotation Types</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  {Object.entries(datasetStats.annotation_types).map(([type, data]) => (
                    <div key={type} className="border rounded-lg p-4">
                      <h4 className="font-semibold text-lg mb-2 capitalize">{type}</h4>
                      <div className="space-y-2">
                        <p>Count: {data.count}</p>
                        {data.pro_percent > 0 && (
                          <div>
                            <p>Pro: {data.pro_percent}%</p>
                            <p>Con: {data.con_percent}%</p>
                          </div>
                        )}
                        <p className="text-sm text-gray-600">{data.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Topic Distribution */}
              <div className="bg-white rounded-lg shadow p-6">
                <h3 className="text-xl font-bold mb-4">Topic Distribution</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">Few-Shot Topics</h4>
                    <ul className="space-y-2">
                      <li>Training: {datasetStats.few_shot.topics.few_shot.train}</li>
                      <li>Development: {datasetStats.few_shot.topics.few_shot.dev}</li>
                      <li>Test: {datasetStats.few_shot.topics.few_shot.test}</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Zero-Shot Topics</h4>
                    <ul className="space-y-2">
                      <li>Training: {datasetStats.few_shot.topics.zero_shot.train}</li>
                      <li>Development: {datasetStats.few_shot.topics.zero_shot.dev}</li>
                      <li>Test: {datasetStats.few_shot.topics.zero_shot.test}</li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'baselines' && (
        <div className="space-y-8">
            <section>
            <BaselineSection />
            </section>
        </div>
        )}

        {activeTab === 'methods' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-6">Methods & Models</h2>
              
              {/* Method Selection Tabs */}
              <div className="flex space-x-4 mb-6">
                {Object.entries(methodDetails).map(([id, method]) => (
                  <button
                    key={id}
                    className={`px-4 py-2 rounded-lg ${
                      activeMethodTab === id 
                      ? 'bg-blue-500 text-white' 
                      : 'bg-gray-100 hover:bg-gray-200'
                    }`}
                    onClick={() => setActiveMethodTab(id)}
                  >
                    {method.name}
                  </button>
                ))}
              </div>

              {/* Method Details */}
              <div className="bg-white rounded-lg shadow p-6">
                <div className="prose max-w-none mb-6">
                  <h3 className="text-2xl font-bold">
                    {methodDetails[activeMethodTab].name}
                  </h3>
                  <p className="text-gray-600">
                    {methodDetails[activeMethodTab].description}
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-6">
                  {methodDetails[activeMethodTab].architecture.map((component, idx) => (
                    <div key={idx} className="border rounded-lg p-6">
                      <h4 className="text-xl font-bold mb-4">{component.component}</h4>
                      
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                          <h5 className="font-semibold mb-2">Features</h5>
                          <ul className="space-y-2">
                            {component.details.map((detail, i) => (
                              <li key={i} className="flex items-start">
                                <span className="text-blue-500 mr-2">•</span>
                                <span>{detail}</span>
                              </li>
                            ))}
                          </ul>
                        </div>
                        
                        <div className="bg-gray-50 rounded-lg p-4">
                          <h5 className="font-semibold mb-2">Technical Details</h5>
                          <p className="text-sm whitespace-pre-line">
                            {component.technical}
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Modifications section for modified TGANet */}
                  {activeMethodTab === 'modified_tganet' && (
                    <div className="border-t mt-6 pt-6">
                      <h4 className="text-xl font-bold mb-4">Modifications & Impact</h4>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        {methodDetails.modified_tganet.modifications.map((mod, idx) => (
                          <div key={idx} className="bg-gray-50 rounded-lg p-4">
                            <h5 className="font-semibold text-blue-600">{mod.change}</h5>
                            <p className="text-sm mt-2"><span className="font-medium">Reason:</span> {mod.reason}</p>
                            <p className="text-sm mt-1"><span className="font-medium">Impact:</span> {mod.impact}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </section>
          </div>
        )}

        {activeTab === 'results' && (
          <div className="space-y-8">
            <section>
              <h2 className="text-2xl font-bold mb-6">Results Analysis</h2>

              {/* 1. TGANet Paper Results */}
              <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h3 className="text-xl font-bold mb-4">1. Original TGANet Results (Allaway & McKeown, 2020)</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">Performance on VAST Dataset</h4>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={Object.entries(combinedResults.paper_results).map(([name, metrics]) => ({
                          name: name,
                          accuracy: metrics.accuracy,
                          f1: metrics.f1
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
                          <Bar dataKey="f1" fill="#82ca9d" name="F1 Score" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="bg-gray-50 p-4 rounded">
                      <h4 className="font-semibold mb-2">Key Findings</h4>
                      <ul className="space-y-2 text-sm">
                        <li>• TGANet achieved best performance (F1: 0.666)</li>
                        <li>• Significant improvement over BERT-joint baseline</li>
                        <li>• Strong performance on unseen topics</li>
                        <li>• Consistent results across different stance types</li>
                      </ul>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <h4 className="font-semibold mb-2">Model Comparison</h4>
                      <div className="text-sm space-y-1">
                        <p>• TGA Net: F1 0.666, Accuracy 0.665</p>
                        <p>• BERT-joint: F1 0.653, Accuracy 0.653</p>
                        <p>• BERT-sep: F1 0.501, Accuracy 0.501</p>
                        <p>• Cross-Net: F1 0.455, Accuracy 0.455</p>
                        <p>• BiCond: F1 0.415, Accuracy 0.415</p>
                        <p>• C-FFNN: F1 0.300, Accuracy 0.300</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 2. Our Implementation Results */}
              <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h3 className="text-xl font-bold mb-4">2. Our Implementation Results</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">Model Performance Comparison</h4>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={Object.entries(combinedResults.our_implementation).map(([name, metrics]) => ({
                          name: name,
                          accuracy: metrics.accuracy,
                          f1: metrics.f1
                        }))}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy" />
                          <Bar dataKey="f1" fill="#82ca9d" name="F1 Score" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Confusion Matrices</h4>
                    <div className="grid grid-cols-2 gap-4">
                      {Object.entries(confusionMatrices).slice(0, 2).map(([model, matrix]) => (
                        <div key={model} className="border rounded-lg p-2">
                          <h5 className="font-medium text-sm mb-2">{model}</h5>
                          <div className="relative overflow-x-auto">
                            <table className="w-full text-xs">
                              <thead className="bg-gray-50">
                                <tr>
                                  <th className="px-1 py-1"></th>
                                  <th className="px-1 py-1">A</th>
                                  <th className="px-1 py-1">D</th>
                                  <th className="px-1 py-1">N</th>
                                </tr>
                              </thead>
                              <tbody>
                                {matrix.map((row, i) => (
                                  <tr key={i} className="border-b">
                                    <th className="px-1 py-1 font-medium bg-gray-50">
                                      {['A', 'D', 'N'][i]}
                                    </th>
                                    {row.map((cell, j) => (
                                      <td key={j} className="px-1 py-1 text-center">
                                        {cell}
                                      </td>
                                    ))}
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="mt-4 bg-gray-50 p-4 rounded">
                      <h4 className="font-semibold mb-2">Implementation Highlights</h4>
                      <ul className="space-y-2 text-sm">
                        <li>• Successfully replicated TGANet performance</li>
                        <li>• TESTED framework showed promising results (F1: 0.594)</li>
                        <li>• Traditional ML baselines provided strong benchmarks</li>
                        <li>• GPT-4 showed limitations in structured stance detection</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>

              {/* 3. Combined Analysis */}
              <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h3 className="text-xl font-bold mb-4">3. Combined Analysis</h3>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h4 className="font-semibold mb-2">Performance Comparison Across All Models</h4>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={[
                          ...Object.entries(combinedResults.paper_results).map(([name, metrics]) => ({
                            name: name,
                            f1: metrics.f1,
                            source: 'Original Paper'
                          })),
                          ...Object.entries(combinedResults.our_implementation).map(([name, metrics]) => ({
                            name: name,
                            f1: metrics.f1,
                            source: 'Our Implementation'
                          }))
                        ]}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                          <YAxis domain={[0, 1]} />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="f1" fill="#82ca9d" name="F1 Score" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="bg-gray-50 p-4 rounded">
                      <h4 className="font-semibold mb-2">Key Observations</h4>
                      <ul className="space-y-2 text-sm">
                        <li>• TGANet maintains superior performance across implementations</li>
                        <li>• TESTED framework shows competitive results</li>
                        <li>• Traditional ML methods provide reliable baselines</li>
                        <li>• LLM approach shows room for improvement</li>
                      </ul>
                    </div>
                    <div className="bg-gray-50 p-4 rounded">
                      <h4 className="font-semibold mb-2">Performance Ranges</h4>
                      <div className="text-sm space-y-1">
                        <p>• Top Tier (F1 {'>'} 0.6): TGANet, BERT-joint</p>
                        <p>• Mid Tier (F1 0.4-0.6): TESTED, BERT-sep, Cross-Net</p>
                        <p>• Lower Tier (F1 {'<'} 0.4): Traditional ML, LLM approaches</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* 4. Ablation Study */}
                <div className="bg-white rounded-lg shadow p-6 mb-8">
                <h3 className="text-xl font-bold mb-4">4. TGANet Parameter Sweep Results</h3>
                <p className="mb-4">
                    We conducted an extensive parameter sweep for our modified TGANet implementation to find the optimal configuration:
                </p>
                <div className="overflow-x-auto">
                    <table className="min-w-full border-collapse border border-gray-200">
                    <thead>
                        <tr>
                        <th className="border border-gray-200 px-4 py-2">Batch Size</th>
                        <th className="border border-gray-200 px-4 py-2">Learning Rate</th>
                        <th className="border border-gray-200 px-4 py-2">Dropout</th>
                        <th className="border border-gray-200 px-4 py-2">Macro F1</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[
                        { batchSize: 16, learningRate: "1e-4", dropout: 0.0, macroF1: 0.324 },
                        { batchSize: 16, learningRate: "1e-4", dropout: 0.2, macroF1: 0.354 },
                        { batchSize: 16, learningRate: "2e-5", dropout: 0.0, macroF1: 0.340 },
                        { batchSize: 16, learningRate: "2e-5", dropout: 0.2, macroF1: 0.334 },
                        { batchSize: 32, learningRate: "1e-4", dropout: 0.0, macroF1: 0.338 },
                        { batchSize: 32, learningRate: "1e-4", dropout: 0.2, macroF1: 0.342 },
                        { batchSize: 32, learningRate: "2e-5", dropout: 0.0, macroF1: 0.335 },
                        { batchSize: 32, learningRate: "2e-5", dropout: 0.2, macroF1: 0.332 },
                        ].map((param, idx) => (
                        <tr
                            key={idx}
                            className={`${
                            param.macroF1 === 0.354 ? "bg-yellow-100" : "bg-white"
                            }`}
                        >
                            <td className="border border-gray-200 px-4 py-2 text-center">
                            {param.batchSize}
                            </td>
                            <td className="border border-gray-200 px-4 py-2 text-center">
                            {param.learningRate}
                            </td>
                            <td className="border border-gray-200 px-4 py-2 text-center">
                            {param.dropout}
                            </td>
                            <td className="border border-gray-200 px-4 py-2 text-center">
                            {param.macroF1.toFixed(3)}
                            </td>
                        </tr>
                        ))}
                    </tbody>
                    </table>
                </div>
                </div>
            </section>
          </div>
        )}

        {activeTab === 'discussion' && (
          <div className="space-y-8">
            <h2 className="text-2xl font-bold mb-6">Discussion & Analysis</h2>

            {/* Baseline Analysis */}
            <section className="bg-white rounded-lg shadow p-6">
              <h3 className="text-xl font-bold mb-4">Traditional & ML Baseline Analysis</h3>
              <div className="mb-6">
                <h4 className="text-lg font-semibold text-blue-800 mb-3">Traditional Approaches</h4>
                <p className="text-gray-700 mb-4">
                  Traditional baselines established clear performance floors, with Random (F1: 0.335) and 
                  Majority Class (F1: 0.169) providing important reference points:
                </p>
                <ul className="list-disc pl-6 space-y-2 mb-4">
                  <li>Random baseline outperforming majority class indicates dataset balance implications</li>
                  <li>Simple heuristics prove insufficient for stance detection complexity</li>
                </ul>
              </div>

              <div className="mb-6">
                <h4 className="text-lg font-semibold text-blue-800 mb-3">ML-Based Methods</h4>
                <p className="text-gray-700 mb-4">
                  Classical ML approaches showed modest improvements:
                </p>
                <ul className="list-disc pl-6 space-y-2 mb-4">
                  <li>TF-IDF + Naive Bayes (F1: 0.343) performed best among traditional ML</li>
                  <li>Random Forest (F1: 0.297) showed limitations in high-dimensional text space</li>
                  <li>Logistic Regression (F1: 0.331) demonstrated competitive linear baseline</li>
                </ul>
              </div>
            </section>

            {/* Model Performance Analysis */}
            <section className="bg-white rounded-lg shadow p-6">
              <h3 className="text-xl font-bold mb-4">Advanced Model Analysis</h3>
              
              <div className="mb-6">
                <h4 className="text-lg font-semibold text-blue-800 mb-3">TGANet Superiority</h4>
                <p className="text-gray-700 mb-4">
                  TGANet's performance (F1: 0.666) significantly surpassed all baselines, showcasing:
                </p>
                <ul className="list-disc pl-6 space-y-2 mb-4">
                  <li>Nearly 2x improvement over best traditional ML baseline</li>
                  <li>Effective topic-grouped attention mechanism</li>
                  <li>Strong generalization to unseen topics</li>
                </ul>
              </div>

              {/* LLM Analysis */}
              <div>
                <h4 className="text-lg font-semibold text-blue-800 mb-3">LLM Performance</h4>
                <p className="text-gray-700 mb-4">
                  GPT-4's underwhelming performance (F1: 0.21) revealed:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h5 className="font-semibold mb-2">Key Findings</h5>
                    <ul className="space-y-2">
                      <li className="flex items-start">
                        <span className="text-red-500 mr-2">•</span>
                        <span>Performed worse than simple baselines</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-red-500 mr-2">•</span>
                        <span>Structured output constraints limited effectiveness</span>
                      </li>
                    </ul>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h5 className="font-semibold mb-2">Implications</h5>
                    <ul className="space-y-2">
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>Specialized architectures outperform general-purpose LLMs</span>
                      </li>
                      <li className="flex items-start">
                        <span className="text-green-500 mr-2">•</span>
                        <span>Task-specific design crucial for stance detection</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>
    </div>
  );
};

export default ProjectPage;