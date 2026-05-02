# Deep NLP Pipeline & LLM Evaluation Report

**Total SOPs Found**: 3

## Document: SOP-IT-001.txt
### 1. NLP Extraction (took 15.01s)
- Language Detected: de
- Primary Domain: IT_OT_security
- Primary Tone: narrative_explanatory
- SHALL Ratio (Raw Signal): 0.0
- Precision Blocklist Built: PASS
- Domain Seeds Generated: 0 seeds
- Step Audit Generated: PASS

### 2. Vector Store Ingestion
- PASS: Ingested 4 chunks into Qdrant using V2 taxonomy.

### 3. LLM Fidelity Execution
#### Mode: `rewrite`
```text
SOP-IT-001 – Zugriffsmanagement auf Produktionsnetzwerk (OT)

**Zweck:** Nur autorisierte Personen (intern/extern/KI) dürfen aufs OT-Netz zugreifen. Die Trennung zwischen IT und OT erfolgt gemäß IEC 62443.

**Geltungsbereich:** Dieses Verfahren gilt für IT-Benutzer, externe Dienstleister, Produktion, Wartung und KI-Systeme.

**Zugriffsarten:** Zugriffe sind in den folgenden Kategorien unterteilt: ...
```
- **Check**: Retained Traceability ID (SOP-IT-001) -> PASS
#### Mode: `improve`
```text
SOP-IT-001 – Zugriffsmanagement auf Produktionsnetzwerk (OT)

**Zweck:**
Nur autorisierte Personen (intern/extern/KI) dürfen aufs OT-Netz zugreifen. Die Trennung zwischen IT und OT erfolgt nach den Vorgaben des IEC 62443.

**Geltungsbereich:**
Dieses Verfahren gilt für IT-Benutzer, externe Dienstleister, Produktion, Wartung und KI-Systeme.

**Zugriffsarten:**
- Read-only
- Operator
- Service
- Adm...
```
#### Mode: `create_new`
```text
SOP-IT-001 – Zugriffsmanagement auf Produktionsnetzwerk (OT)

### Zweck
Der Zweck dieser Standardoperative Prozedur (SOP) ist es, sicherzustellen, dass nur autorisierte Personen (intern/extern/KI) auf das Produktionsnetzwerk (OT) zugreifen dürfen. Die Trennung zwischen IT- und OT-Netzwerk nach den Vorgaben des IEC 62443 Standards gewährleistet eine sichere Infrastruktur.

### Geltungsbereich
Diese...
```

---

## Document: SOP-IT-002.txt
### 1. NLP Extraction (took 1.97s)
- Language Detected: de
- Primary Domain: IT_OT_security
- Primary Tone: narrative_explanatory
- SHALL Ratio (Raw Signal): 0.0
- Precision Blocklist Built: PASS
- Domain Seeds Generated: 0 seeds
- Step Audit Generated: PASS

### 2. Vector Store Ingestion
- PASS: Ingested 4 chunks into Qdrant using V2 taxonomy.

### 3. LLM Fidelity Execution
#### Mode: `rewrite`
```text
SOP-IT-002 – Netzwerksicherheit & Firewall (OT/IT-Trennung)

**Zweck:** Schutz des Produktionsnetzwerks vor Büro- & Internetzugriffen.

**Geltungsbereich:** Firewall IT/OT, Segmentierung (Fermentation, Aufreinigung, Abfüllung), Remote-Zugänge, Produktions-WLAN.

**Key Points:**

- **Firewall-Regeln:** IT→OT nur Port 443 (Read-only) und Port 22 (nur Admin mit 2FA).
- **Internet → OT:** komplett blo...
```
- **Check**: Retained Traceability ID (SOP-IT-002) -> PASS
#### Mode: `improve`
```text
SOP-IT-002 – Netzwerksicherheit & Firewall (OT/IT-Trennung)

**Zweck:**
Schutz des Produktionsnetzwerks vor Büro- & Internetzugriffen.

**Geltungsbereich:**
Firewall IT/OT, Segmentierung (Fermentation, Aufreinigung, Abfüllung), Remote-Zugänge, Produktions-WLAN.

**Key Points:**

- **Firewall-Regeln:** IT→OT nur 443 (Read-only) & 22 (nur Admin mit 2FA).
- **Internet → OT:** komplett blockiert.
- **...
```
#### Mode: `create_new`
```text
SOP-IT-002 – Netzwerksicherheit & Firewall (OT/IT-Trennung)

### Zweck
Der Zweck dieser Standardoperative Prozedur (SOP) ist es, eine effektive und sichere Vorgehensweise für den Notfallabbau im Produktionsnetzwerk zu definieren, um die Netzwerksicherheit und die Trennung zwischen IT und OT zu gewährleisten.

### Geltungsbereich
Diese Prozedur gilt für alle Mitarbeiter, die mit dem Produktionsnetz...
```

---

## Document: SOP-IT-003.txt
### 1. NLP Extraction (took 0.57s)
- Language Detected: de
- Primary Domain: QMS_general
- Primary Tone: narrative_explanatory
- SHALL Ratio (Raw Signal): 0.0
- Precision Blocklist Built: PASS (None Required)
- Domain Seeds Generated: 0 seeds
- Step Audit Generated: PASS

### 2. Vector Store Ingestion
- PASS: Ingested 6 chunks into Qdrant using V2 taxonomy.

### 3. LLM Fidelity Execution
#### Mode: `rewrite`
```text
SOP-IT-003 – Notfallzugriff (Break-Glass-Verfahren)

**SCOPE**

Geltungsbereich: Stillstand >30 Min, Ausfall Authentifizierung, IT-Admin nicht verfügbar.

**PROCEDURE**

Aktivierung: Beide physischen YubiKey-Token gemeinsam im Terminal → max. 2 Stunden Zugriff.

Nachbereitung in 24h: Dokumentation, regulären Zugriff beantragen, neue Token erstellen.

**RECORDS**

AUD-IT-013 – EMA: Break-Glass ohne...
```
- **Check**: Retained Traceability ID (SOP-IT-003) -> PASS
#### Mode: `improve`
```text
SOP-IT-003 – Notfallzugriff (Break-Glass-Verfahren)

**SCOPE**

Geltungsbereich: Stillstand >30 Minuten, Ausfall der Authentifizierung, IT-Admin nicht verfügbar.

**PROCEDURE**

Aktivierung: Beide physischen YubiKey-Token gemeinsam im Terminal einzugeben → maximal 2 Stunden Zugriff.

Nachbereitung in 24 Stunden: Dokumentation der Nutzung, Antrag auf regulären Zugriff stellen, neue Token erstellen....
```
#### Mode: `create_new`
```text
SOP-IT-003 – Notfallzugriff (Break-Glass-Verfahren)

### ZWECK
Der Zweck dieses Verfahrens ist der temporäre Zugriff auf die Operative Technik (OT) bei kritischen Produktionsstillständen, wenn regulärer Zugang nicht möglich ist.

### GELTUNGSBEREICH
Dieses Verfahren gilt für Stillstände, die über 30 Minuten anhalten, bei denen die Authentifizierung fehlschlägt, und bei denen der IT-Admin nicht ver...
```

---
