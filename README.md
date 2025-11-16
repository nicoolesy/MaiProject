# Gemini AI Parenting Assistant

### Highlights

- **Smart Parenting Advisor**: An AI-powered conversational tool that combines Google's Gemini Pro language model with a custom parenting knowledge database to provide immediate, expert-level guidance for real parenting challenges.
- **Action-Oriented Solutions**: Instead of generic comfort, delivers specific step-by-step strategies tailored to your child's age group (0-18) and situation type, giving parents concrete techniques, timing recommendations, and follow-up plans they can implement right away.
- **Advanced RAG Technology**: Uses retrieval-augmented generation to merge conversational AI intelligence with curated child development expertise, creating a 24/7 accessible parenting consultant that transforms research-backed knowledge into immediately usable action plans.

## What This Is

This is a **smart parenting advisor** that functions like having an expert child psychologist available 24/7 through your computer. Unlike tools that offer generic comfort like "everything will be okay," this system delivers **specific, actionable steps** parents can immediately implement with their children.

## How It Works

The system combines two powerful technologies: a **custom database** I built containing real parenting strategies, developmental milestones, and proven techniques, plus Google's **Gemini AI**—a sophisticated language model that understands complex questions and generates human-like responses. When a parent asks, "My 4-year-old won't listen," the system first searches my database for age-appropriate discipline strategies, then uses the AI to craft a personalized response with concrete steps like "Try the 1-2-3 counting method, set up a reward chart, and implement consistent 4-minute timeouts."

## What Makes It Different

This tool acts as a **practical parenting coach** rather than just offering emotional support. It considers your child's specific age (since a 3-year-old and a 13-year-old need completely different approaches) and the type of challenge you're facing (behavioral issues require different strategies than emotional problems). The AI goes beyond saying "be patient"—it provides step-by-step instructions, suggests specific phrases to use, recommends timing for interventions, and offers follow-up strategies if the first approach doesn't work.

## The Real Innovation

I've created a system that transforms general parenting knowledge into **immediately usable action plans** tailored to your specific situation. This gives parents the confidence to handle challenges with proven, research-backed methods instead of guessing what might work.

> _“Ask me anything about parenting — I’ll respond like a calm, supportive coach.”_

An interactive AI parenting coach powered by **Google Gemini 2.0 Flash Lite**, built with **Streamlit**, **FastAPI**, and **ChromaDB**.  
It provides personalized, empathetic guidance for parents — tailored by a child’s **age group**, **behavior topic**, and conversation tone.

---

## 🌟 Demo

![App Demo](demo.png)  
_(Example question: “My kid doesn’t like to go to school.”)_

---

## Features

- Age- and topic-aware parenting answers  
- Real-time AI replies using **Gemini 2.0 Flash Lite**  
- Emotional tone adaptation via **VADER Sentiment Analysis**  
- Cognitive-intent tuning using **Bloom’s Taxonomy**  
- Local knowledge retrieval with **ChromaDB**  
- Seamless backend powered by **FastAPI + Render**

---

## Tech Stack Overview

| Layer          | Tools                                         |
| -------------- | --------------------------------------------- |
| **Frontend**   | Streamlit                                     |
| **Backend**    | FastAPI                                       |
| **Model**      | Google Gemini 2.0 Flash Lite                  |
| **Database**   | ChromaDB                                      |
| **NLP**        | VADER Sentiment Analyzer                      |
| **Env Mgmt**   | dotenv                                        |
| **Deployment** | Render (backend) + Streamlit Cloud (frontend) |

---

## Deployment

| **Component**       | **Platform**                                    |
| ------------------- | ----------------------------------------------- |
| **Backend**         | [Render ↗](https://render.com)                  |
| **Frontend**        | [Streamlit Cloud ↗](https://streamlit.io/cloud) |
| **Environment Var** | `GOOGLE_API_KEY` — set in Render Settings       |

---

## Performance Tips

 💡 Use "gemini-2.0-flash-lite" for fastest responses

 💡 Keep prompt context under ≈ 500 characters

 💡 Run Chroma locally for low-latency retrieval

## Author

**Nicole Chen**  
AI & Data Science | University of Michigan
[LinkedIn ↗](https://www.linkedin.com/in/nicoolesy) | [GitHub ↗](https://github.com/nicoolesy)

> 🧡 _Built to help parents understand, connect, and grow together — one gentle conversation at a time._
