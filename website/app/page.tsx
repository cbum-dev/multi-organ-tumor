'use client';
import React, { useState, useEffect } from 'react';
import { Camera, Brain, Waves, Activity, Shield, Zap, ChevronRight, Sparkles, BarChart3, Clock, Users, FileText } from 'lucide-react';
import { SparklesCore } from '@/components/ui/sparkles';

const OncoVisionLanding = () => {
  const [scrolled, setScrolled] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 4);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const features = [
    {
      icon: <Brain className="w-6 h-6" />,
      title: "Multi-Organ Detection",
      description: "Brain, Lung, Liver & Kidney analysis with YOLOv8",
      color: "from-blue-500 to-cyan-500"
    },
    {
      icon: <Sparkles className="w-6 h-6" />,
      title: "Advanced AI Ensemble",
      description: "Monte Carlo uncertainty & radiomics extraction",
      color: "from-purple-500 to-pink-500"
    },
    {
      icon: <BarChart3 className="w-6 h-6" />,
      title: "Temporal Analytics",
      description: "Track tumor growth and progression over time",
      color: "from-orange-500 to-red-500"
    },
    {
      icon: <Shield className="w-6 h-6" />,
      title: "Clinical Intelligence",
      description: "Differential diagnosis & treatment protocols",
      color: "from-green-500 to-emerald-500"
    }
  ];

  const stats = [
    { icon: <Activity className="w-5 h-5" />, label: "Detection Accuracy", value: "98.5%" },
    { icon: <Clock className="w-5 h-5" />, label: "Analysis Time", value: "<3s" },
    { icon: <Users className="w-5 h-5" />, label: "Organs Supported", value: "4+" },
    { icon: <FileText className="w-5 h-5" />, label: "Features Extracted", value: "15+" }
  ];

  const capabilities = [
    "CLAHE Medical-Grade Preprocessing",
    "5-Dimensional Quality Assessment",
    "Uncertainty Quantification",
    "Radiomics Feature Extraction",
    "Growth Rate Calculation",
    "Batch Processing",
    "Comprehensive Audit Trail",
    "HIPAA-Compliant Storage"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white overflow-hidden">
      {/* Animated Background */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      {/* Navigation */}
      <nav className={`fixed top-0 w-full z-50 transition-all duration-300 ${scrolled ? 'bg-slate-950/90 backdrop-blur-xl border-b border-blue-500/20' : ''}`}>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-20">
            <div className="flex items-center space-x-3">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg blur-lg opacity-50"></div>
                <div className="relative bg-gradient-to-r from-blue-600 to-cyan-600 p-2 rounded-lg">
                  <Waves className="w-7 h-7" />
                </div>
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  OncoVision AI
                </h1>
                <p className="text-xs text-blue-300/70">Ultra Edition v4.0</p>
              </div>
            </div>
            
            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-slate-300 hover:text-white transition-colors">Features</a>
              <a href="#capabilities" className="text-slate-300 hover:text-white transition-colors">Capabilities</a>
              <a href="#about" className="text-slate-300 hover:text-white transition-colors">About</a>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      {/* <SparklesCore/> */}
      <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center space-y-8 mb-16">
            <div className="inline-flex items-center space-x-2 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 border border-blue-400/30 rounded-full px-6 py-2.5 backdrop-blur-sm animate-pulse">
              <Sparkles className="w-4 h-4 text-blue-300" />
              <span className="text-sm font-medium bg-gradient-to-r from-blue-300 to-cyan-300 bg-clip-text text-transparent">
                AI-Powered Medical Imaging Platform
              </span>
            </div>
            
            <h2 className="text-5xl md:text-7xl font-bold leading-tight">
              <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
                Revolutionizing
              </span>
              <br />
              <span className="text-white">Cancer Diagnosis & Care</span>
            </h2>
            
            <p className="text-xl text-slate-300 max-w-3xl mx-auto leading-relaxed">
              Advanced AI technology for precise tumor detection across multiple organs, 
              providing clinicians with powerful tools for early diagnosis and treatment planning.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-6">
              <a 
                href="http://localhost:7860" 
                target="_blank" 
                rel="noopener noreferrer"
                className="group relative inline-flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 px-8 py-4 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-blue-500/50 hover:shadow-xl hover:shadow-blue-500/70 hover:scale-105"
              >
                <Zap className="w-5 h-5 text-yellow-300" />
                <span>Launch Live Demo</span>
                <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </a>
              
              <a 
                href="#features" 
                className="group inline-flex items-center space-x-2 bg-transparent hover:bg-slate-800/50 border-2 border-slate-700 hover:border-blue-500/50 px-8 py-3.5 rounded-xl font-medium transition-all duration-300"
              >
                <Activity className="w-5 h-5 text-blue-400" />
                <span>Explore Features</span>
              </a>
              
              <button className="group inline-flex items-center space-x-2 bg-slate-900/80 hover:bg-slate-800/90 border border-slate-700/50 hover:border-slate-600/70 px-6 py-3.5 rounded-xl font-medium transition-all duration-300">
                <FileText className="w-5 h-5 text-slate-300 group-hover:text-blue-400" />
                <span>View Docs</span>
              </button>
            </div>
            
            <div className="flex flex-wrap items-center justify-center gap-3 pt-4 text-sm text-slate-400">
              <div className="flex items-center">
                <span className="flex w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                <span>Real-time Analysis</span>
              </div>
              <div className="hidden sm:block">•</div>
              <div className="flex items-center">
                <span className="flex w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
                <span>Multi-Organ Support</span>
              </div>
              <div className="hidden sm:block">•</div>
              <div className="flex items-center">
                <span className="flex w-2 h-2 bg-cyan-500 rounded-full mr-2"></span>
                <span>HIPAA Compliant</span>
              </div>
            </div>

            {/* Stats Bar */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-12 max-w-4xl mx-auto">
              {stats.map((stat, i) => (
                <div key={i} className="bg-slate-800/30 backdrop-blur-sm border border-slate-700/50 rounded-xl p-4 hover:bg-slate-800/50 transition-all duration-300 hover:scale-105">
                  <div className="flex items-center justify-center mb-2 text-blue-400">
                    {stat.icon}
                  </div>
                  <div className="text-2xl font-bold text-white">{stat.value}</div>
                  <div className="text-xs text-slate-400">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h3 className="text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Revolutionary Features
              </span>
            </h3>
            <p className="text-slate-400 text-lg">
              Cutting-edge AI capabilities for comprehensive oncological analysis
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, i) => (
              <div 
                key={i}
                className={`group relative bg-slate-800/30 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 hover:bg-slate-800/50 transition-all duration-500 hover:scale-105 cursor-pointer ${activeFeature === i ? 'ring-2 ring-blue-500' : ''}`}
                onMouseEnter={() => setActiveFeature(i)}
              >
                <div className={`absolute inset-0 bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity duration-500`}></div>
                
                <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br ${feature.color} mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  {feature.icon}
                </div>
                
                <h4 className="text-xl font-bold mb-2 text-white group-hover:text-blue-300 transition-colors">
                  {feature.title}
                </h4>
                <p className="text-slate-400 text-sm leading-relaxed">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Capabilities Grid */}
      <section id="capabilities" className="relative py-20 px-4 sm:px-6 lg:px-8 bg-slate-900/50">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h3 className="text-4xl font-bold mb-4">
              <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                Complete Diagnostic Suite
              </span>
            </h3>
            <p className="text-slate-400 text-lg">
              Enterprise-grade medical imaging analysis
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
            {capabilities.map((capability, i) => (
              <div 
                key={i}
                className="flex items-center space-x-3 bg-slate-800/30 backdrop-blur-sm border border-slate-700/50 rounded-lg p-4 hover:bg-slate-800/50 hover:border-blue-500/50 transition-all duration-300 hover:scale-105"
              >
                <div className="flex-shrink-0 w-2 h-2 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"></div>
                <span className="text-sm text-slate-300">{capability}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technology Stack */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="bg-gradient-to-br from-slate-800/50 to-blue-900/30 backdrop-blur-sm border border-slate-700/50 rounded-3xl p-12">
            <div className="grid md:grid-cols-2 gap-12 items-center">
              <div>
                <h3 className="text-3xl font-bold mb-6">
                  <span className="bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                    Powered by Advanced AI
                  </span>
                </h3>
                <div className="space-y-4">
                  <div className="flex items-start space-x-3">
                    <Zap className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold text-white mb-1">YOLOv8 Detection Engine</h4>
                      <p className="text-sm text-slate-400">Real-time multi-organ tumor detection with state-of-the-art accuracy</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Brain className="w-5 h-5 text-purple-400 flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold text-white mb-1">Monte Carlo Uncertainty</h4>
                      <p className="text-sm text-slate-400">Quantify prediction confidence with dropout-based variance estimation</p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-3">
                    <Activity className="w-5 h-5 text-green-400 flex-shrink-0 mt-1" />
                    <div>
                      <h4 className="font-semibold text-white mb-1">Radiomics Extraction</h4>
                      <p className="text-sm text-slate-400">15+ quantitative imaging biomarkers for comprehensive tumor characterization</p>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl blur-2xl"></div>
                <div className="relative bg-slate-900/80 backdrop-blur-sm border border-slate-700 rounded-2xl p-8">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Model Architecture</span>
                      <span className="text-blue-400 font-mono">YOLOv8n</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Preprocessing</span>
                      <span className="text-green-400 font-mono">CLAHE</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Feature Extraction</span>
                      <span className="text-purple-400 font-mono">15+ metrics</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400">Confidence Threshold</span>
                      <span className="text-yellow-400 font-mono">Adaptive</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <div className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 backdrop-blur-sm border border-blue-500/30 rounded-3xl p-12">
            <h3 className="text-4xl font-bold mb-4">
              Ready to Experience
              <br />
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Next-Gen Medical AI?
              </span>
            </h3>
            <p className="text-slate-400 text-lg mb-8">
              Launch the OncoVision AI Ultra scanner and start analyzing medical images with unprecedented precision.
            </p>
            <a 
              href="http://localhost:7860" 
              target="_blank" 
              rel="noopener noreferrer"
              className="group inline-flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 px-10 py-5 rounded-xl font-semibold text-lg transition-all duration-300 shadow-2xl shadow-blue-500/50 hover:shadow-blue-500/70 hover:scale-105"
            >
              <Camera className="w-6 h-6" />
              <span>Launch Scanner Now</span>
              <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
            </a>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative border-t border-slate-800 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="grid md:grid-cols-3 gap-8 mb-8">
            <div>
              <h4 className="text-lg font-semibold mb-4 text-white">OncoVision AI Ultra</h4>
              <p className="text-slate-400 text-sm leading-relaxed">
                Advanced medical imaging AI platform for comprehensive oncological analysis and diagnosis.
              </p>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4 text-white">Features</h4>
              <ul className="space-y-2 text-sm text-slate-400">
                <li>Multi-Organ Detection</li>
                <li>Radiomics Analysis</li>
                <li>Temporal Tracking</li>
                <li>Quality Assessment</li>
              </ul>
            </div>
            <div>
              <h4 className="text-lg font-semibold mb-4 text-white">Legal</h4>
              <p className="text-xs text-slate-500 leading-relaxed">
                FOR RESEARCH & EDUCATIONAL USE ONLY. Not approved for direct clinical use. 
                All findings must be reviewed by qualified medical professionals.
              </p>
            </div>
          </div>
          <div className="border-t border-slate-800 pt-8 text-center text-sm text-slate-500">
            <p>© 2024 OncoVision AI Ultra v4.0. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default OncoVisionLanding;