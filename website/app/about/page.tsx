import React from 'react';
import { Brain, Users, Award, HeartPulse, Shield, Zap } from 'lucide-react';
import Link from 'next/link';

const team = [
  {
    name: 'Dr. Sarah Chen',
    role: 'Chief Medical Officer',
    bio: 'Board-certified radiologist with 15+ years of experience in medical imaging and AI applications.',
    image: '/team/sarah.jpg'
  },
  {
    name: 'Michael Rodriguez',
    role: 'Lead AI Researcher',
    bio: 'PhD in Computer Vision with expertise in deep learning for medical image analysis.',
    image: '/team/michael.jpg'
  },
  {
    name: 'Priya Patel',
    role: 'Product Manager',
    bio: 'Healthcare technology specialist focused on creating intuitive medical software solutions.',
    image: '/team/priya.jpg'
  },
  {
    name: 'David Kim',
    role: 'Software Architect',
    bio: 'Full-stack developer with a passion for building scalable healthcare applications.',
    image: '/team/david.jpg'
  }
];

const stats = [
  { value: '98.5%', label: 'Detection Accuracy', icon: Award },
  { value: '10,000+', label: 'Scans Processed', icon: HeartPulse },
  { value: '4', label: 'Organs Supported', icon: Brain },
  { value: '100%', label: 'HIPAA Compliant', icon: Shield }
];

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-900 text-white">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-4xl md:text-6xl font-bold mb-6">
            <span className="bg-gradient-to-r from-blue-400 via-cyan-400 to-purple-400 bg-clip-text text-transparent">
              About OncoVision AI
            </span>
          </h1>
          <p className="text-xl text-slate-300 max-w-3xl mx-auto mb-12">
            Revolutionizing cancer diagnosis through advanced AI-powered medical imaging analysis.
          </p>
        </div>
      </section>

      {/* Mission Section */}
      <section className="py-16 bg-slate-900/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="lg:text-center">
            <h2 className="text-3xl font-bold text-white sm:text-4xl mb-8">
              Our Mission
            </h2>
            <p className="mt-4 max-w-4xl text-xl text-slate-300 lg:mx-auto">
              At OncoVision AI, we're committed to transforming cancer care by providing clinicians with powerful, 
              AI-driven tools for early and accurate tumor detection. Our mission is to improve patient outcomes 
              through cutting-edge technology that enhances diagnostic confidence and efficiency.
            </p>
          </div>

          <div className="mt-16 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-6 text-center">
                <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-blue-500/10 text-blue-400 mb-4">
                  <stat.icon className="h-8 w-8" aria-hidden="true" />
                </div>
                <h3 className="text-2xl font-bold text-white">{stat.value}</h3>
                <p className="mt-1 text-slate-400">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-white">Meet Our Team</h2>
            <p className="mt-4 text-xl text-slate-300 max-w-3xl mx-auto">
              A diverse team of medical professionals, AI researchers, and software engineers 
              dedicated to advancing cancer diagnosis.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {team.map((member, index) => (
              <div key={index} className="bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl overflow-hidden hover:shadow-xl hover:shadow-blue-500/10 transition-all duration-300">
                <div className="h-48 bg-slate-700/50 flex items-center justify-center text-slate-400">
                  <Users className="h-20 w-20" />
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-bold text-white">{member.name}</h3>
                  <p className="text-blue-400 mb-3">{member.role}</p>
                  <p className="text-slate-400 text-sm">{member.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-r from-blue-900/30 to-cyan-900/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-6">Ready to experience the future of cancer diagnosis?</h2>
          <p className="text-xl text-slate-300 mb-8 max-w-3xl mx-auto">
            Join leading healthcare providers who trust OncoVision AI for accurate and efficient tumor detection.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <Link
              href="/contact"
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-500 hover:to-cyan-500 rounded-xl font-semibold text-white transition-all duration-300 shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/50"
            >
              <Zap className="w-5 h-5 mr-2" />
              Get Started
            </Link>
            <Link
              href="/features"
              className="inline-flex items-center px-8 py-4 bg-slate-800/50 hover:bg-slate-800/70 border border-slate-700/50 rounded-xl font-medium transition-all duration-300"
            >
              Learn More
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
