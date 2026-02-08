# Project-SightX
AI-powered diabetic retinopathy detection system using transfer learning (ResNet50) to enable early screening and prevent blindness. Built with React, Node.js, PostgreSQL, and PyTorch. A personal mission to combine my passion for biology and technology while honoring my family's journey with diabetes.

## The Story Behind This Project
Diabetes runs in my family. I've watched my relatives navigate the challenges of this disease, including the constant worry about complications like diabetic retinopathy—a leading cause of blindness that often shows no symptoms until it's too late.

As a software engineer with a deep fascination for biology, I've always dreamed of working at the intersection of healthcare and technology. This project represents that convergence: applying my fullstack development skills and diving deep into machine learning to build something that could genuinely help people.

## Project Goals

### Technical Objectives
- **Understand Transfer Learning**: Apply ResNet50 (pre-trained on ImageNet) with custom classification head for medical image analysis
- **Refine Fullstack Architecture**: Build a microservices system with React, Node.js, PostgreSQL, and FastAPI
- **Optimize for M4 Neural Engine**: Leverage Apple Silicon for accelerated inference pipelines
- **Deployment**: Containerize with Docker and design for scalability

### Impact Objectives
- **Serve Underserved Communities**: Bring diagnostic tools to rural areas lacking eye care specialists  
- **Early Detection**: Catch diabetic retinopathy at mild stages (Grade 1-2) before irreversible damage
- **Reduce Healthcare Burden**: Automate initial screening to reserve specialist time for high-risk cases
- **Democratize Screening**: Enable primary care clinics to perform DR screening without specialized ophthalmologists

## 🏗️ System Architecture
```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐      ┌──────────────┐
│    React    │ ───> │   Node.js    │ ───> │ Python FastAPI  │ ───> │  PostgreSQL  │
│  Frontend   │ <─── │   Backend    │ <─── │  ML Inference   │ <─── │   Database   │
│             │      │              │      │   (ResNet50)    │      │   (Records)  │
└─────────────┘      └──────────────┘      └─────────────────┘      └──────────────┘
```

## Why This Matters

**The Problem:**
- 463 million people worldwide have diabetes
- 1 in 3 will develop diabetic retinopathy
- Most won't know until vision loss is permanent
- Shortage of ophthalmologists, especially in rural areas

**The Solution:**
- Automated screening at primary care level
- Scalable to underserved regions
- Early detection when treatment is most effective
- Cost reduction vs treating advanced blindness

## Personal Note

This project represents the intersection of everything I care about: my family's health, my love for biology, and my skills as a software engineer. Every line of code is written with the hope that one day, technology like this could help someone keep their sight.

If you're a healthcare professional, ML researcher, or just someone who believes in using technology for good—I'd love to hear your feedback.

**Built with care and to serve people by Bibesh Timalsina**  
Software Engineer | Advocate for Health Tech | Biology Enthusiast 

## License
MIT License - feel free to use this for learning, research, or building upon.
