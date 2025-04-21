# PaliGemma-Image-Segmentation

An app for performing image segmentation with PaliGemma 2 mix

---

- transformers, JAX/Flax -> PaliGemma 2 mix
- Docker
- FastAPI

---

Structure:

```
project_folder/
├── app/
│   ├── main.py
│   └── segmentation.py
├── models/
│   └── vae-oid.npz
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

---

Workflow Overview:

```mermaid
graph LR
    User([User]) -->|Uploads Image| Client[Client Application]
    User -->|Provides Prompt| Client
    
    Client -->|HTTP POST Request| API[FastAPI Service]
    API -->|Process Image| PaliGemma[PaliGemma Model]
    PaliGemma -->|Generate Segmentation| VAE[VAE Model]
    VAE -->|Create Masks| API
    
    API -->|JSON Response| Client
    Client -->|Display Results| User
    
    style User fill:#f9d5e5,stroke:#d64161,stroke-width:2px
    style Client fill:#eeeeee,stroke:#333333
    style API fill:#b5e7a0,stroke:#86af49
    style PaliGemma fill:#b8e0d2,stroke:#6a9c89
    style VAE fill:#d0e1f9,stroke:#7fa6bc
```

App Architecture:

```mermaid
graph TD
    subgraph "Docker Container"
        subgraph "app/"
            main[main.py
            FastAPI Application]
            segmentation[segmentation.py
            Image Segmentation Logic]
            main -->|imports| segmentation
        end
        
        subgraph "External Dependencies"
            NP[numpy]
            TR[transformers]
            TF[TensorFlow]
            PT[PyTorch]
            JF[JAX/Flax]
        end
        
        subgraph "Models"
            PaliGemma[PaliGemma 2 mix]
            VAE[VAE Checkpoint]
        end
        
        segmentation -->|uses| TR
        segmentation -->|uses| TF
        segmentation -->|uses| PT
        segmentation -->|uses| JF
        segmentation -->|uses| NP
        
        main -->|loads| PaliGemma
        segmentation -->|loads| VAE
    end
    
    Client[Client Application] -->|HTTP Requests| main
    
    subgraph "API Endpoints"
        segment[POST /segment/]
        root[GET /]
    end
    
    main -->|defines| segment
    main -->|defines| root
    Client -->|calls| segment
    Client -->|calls| root
    
    style Docker fill:#e7f4ff,stroke:#0078d7
    style main fill:#c2e0ff,stroke:#0078d7
    style segmentation fill:#c2e0ff,stroke:#0078d7
    style Client fill:#ffd7b5,stroke:#ff8c00
    style segment fill:#d5e8d4,stroke:#82b366
    style root fill:#d5e8d4,stroke:#82b366
```
