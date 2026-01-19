import hashlib
import json
from typing import Dict, Any, Optional
from functools import lru_cache
import ollama
from openai import OpenAI
import os

from src.pipeline.prompt import VALID_LABELS_DICT, VALID_LABELS_STR, PROMPT_V3


class LLMCache:
   """Cache in-memory con chiave basata su hash del contesto"""
   
   def __init__(self, max_size: int = 1000):
      self.cache: Dict[str, Dict[str, Any]] = {}
      self.max_size = max_size
      self.hits = 0
      self.misses = 0
   
   def _make_key(self, target: str, context: str, prev_label: str, current_pred: str) -> str:
      """Crea chiave hash univoca per il contesto"""
      content = f"{target}|{context}|{prev_label}|{current_pred}"
      return hashlib.md5(content.encode()).hexdigest()
   
   def get(self, target: str, context: str, prev_label: str, current_pred: str) -> Optional[Dict[str, Any]]:
      """Recupera risultato dalla cache"""
      key = self._make_key(target, context, prev_label, current_pred)
      if key in self.cache:
         self.hits += 1
         return self.cache[key]
      self.misses += 1
      return None
   
   def set(self, target: str, context: str, prev_label: str, current_pred: str, result: Dict[str, Any]):
      """Salva risultato in cache (con limite dimensione)"""
      if len(self.cache) >= self.max_size:
         # Rimuovi il primo elemento (FIFO semplice)
         first_key = next(iter(self.cache))
         del self.cache[first_key]
      
      key = self._make_key(target, context, prev_label, current_pred)
      self.cache[key] = result
   
   def get_stats(self) -> Dict[str, Any]:
      """Statistiche cache"""
      total = self.hits + self.misses
      hit_rate = (self.hits / total * 100) if total > 0 else 0
      return {
         "hits": self.hits,
         "misses": self.misses,
         "hit_rate": f"{hit_rate:.2f}%",
         "size": len(self.cache)
      }


# ============================================================================
# LLM ROUTER OTTIMIZZATO
# ============================================================================
class OptimizedLLMRouter:
   """Router LLM con caching, validazione robusta e gestione errori"""
   
   def __init__(self, 
               source: str = "openai",
               api_key: Optional[str] = None,
               model: str = "gpt-4o-mini",
               ollama_model: str = "qwen2.5:3b",
               enable_cache: bool = True,
               cache_size: int = 1000,
               valid_labels: Optional[set] = None):
      
      self.source = source.lower()
      self.model = model if source == "openai" else ollama_model
      self.cache = LLMCache(max_size=cache_size) if enable_cache else None
      self.valid_labels = valid_labels or self._default_valid_labels()
      
      # Inizializza client
      if self.source == "openai":
         api_key = api_key or os.getenv("OPENAI_API_KEY")
         if not api_key:
               raise ValueError("OpenAI API key required when source='openai'")
         self.client = OpenAI(api_key=api_key)
         print(f"   [LLM] Backend: OpenAI ({self.model})")
      elif self.source == "ollama":
         self.client = None
         print(f"   [LLM] Backend: Ollama ({self.model})")
      else:
         raise ValueError(f"Invalid LLM source: {source}. Use 'openai' or 'ollama'")
   
   def _default_valid_labels(self) -> set:
      """Set di tutti i tag validi (da VALID_LABELS_DICT)"""
      return {
         "O",
         "B-GIVENNAME", "I-GIVENNAME", 
         "B-SURNAME", "I-SURNAME", 
         "B-TITLE", "I-TITLE",
         "B-CITY", "I-CITY", 
         "B-STREET", "I-STREET", 
         "B-BUILDINGNUM", "I-BUILDINGNUM",
         "B-ZIPCODE", "I-ZIPCODE", 
         "B-IDCARDNUM", "I-IDCARDNUM", 
         "B-PASSPORTNUM", "I-PASSPORTNUM",
         "B-DRIVERLICENSENUM", "I-DRIVERLICENSENUM", 
         "B-SOCIALNUM", "I-SOCIALNUM",
         "B-TAXNUM", "I-TAXNUM",
         "B-CREDITCARDNUMBER", "I-CREDITCARDNUMBER",
         "B-EMAIL", "I-EMAIL",
         "B-TELEPHONENUM", "I-TELEPHONENUM",
         "B-DATE", "I-DATE", 
         "B-TIME", "I-TIME", 
         "B-AGE", "I-AGE", 
         "B-SEX", "I-SEX", 
         "B-GENDER", "I-GENDER"
      }
   
   def disambiguate(self,
                  target_token: str,
                  full_text: str,
                  char_start: int,
                  char_end: int,
                  current_pred: str,
                  prev_label: str,
                  lang: str = "en",
                  valid_labels_str: str = "") -> Dict[str, Any]:
      """
      Disambigua un token usando LLM con caching intelligente
      
      Returns:
         Dict con: is_pii (bool), corrected_label (str), reasoning (str), cached (bool)
      """
      
      # 1. Estrai contesto ottimizzato
      context = self._extract_context(full_text, char_start, char_end)
      clean_token = full_text[char_start:char_end].strip()
      
      # 2. Controlla cache
      if self.cache:
         cached_result = self.cache.get(clean_token, context, prev_label, current_pred)
         if cached_result:
               cached_result["cached"] = True
               return cached_result
      
      # 3. Prepara prompt
      try:
         prompt = PROMPT_V3.format(
               context=context,
               target_token=clean_token,
               prev_label=prev_label,
               current_pred=current_pred,
               valid_labels_str=VALID_LABELS_STR)
      except KeyError as e:
         return self._error_response(current_pred, f"Prompt formatting error: {e}")
      
      # 4. Chiama LLM
      try:
         raw_result = self._call_llm(prompt)
         validated_result = self._validate_response(raw_result, current_pred)
         
         # 5. Salva in cache
         if self.cache:
               self.cache.set(clean_token, context, prev_label, current_pred, validated_result)
         
         validated_result["cached"] = False
         return validated_result
         
      except Exception as e:
         print(f"   [LLM ERROR]: {e}")
         return self._error_response(current_pred, str(e))
   
   def _extract_context(self, text: str, start: int, end: int, window: int = 200) -> str:
      """Estrae contesto ottimizzato attorno al token"""
      
      # Context sinistro
      ctx_start = max(0, start - window)
      if ctx_start > 0:
         while ctx_start > 0 and text[ctx_start] not in " \n.":
               ctx_start -= 1
         ctx_start += 1
      
      # Context destro
      ctx_end = min(len(text), end + window)
      if ctx_end < len(text):
         while ctx_end < len(text) and text[ctx_end] not in " \n.":
               ctx_end += 1
      
      prefix = text[ctx_start:start].replace("\n", " ")
      target = text[start:end]
      suffix = text[end:ctx_end].replace("\n", " ")
      
      return f"...{prefix}>>> {target} <<<{suffix}..."
   
   def _call_llm(self, prompt: str) -> Dict[str, Any]:
      """Dispatcher per chiamate LLM"""
      if self.source == "openai":
         return self._call_openai(prompt)
      else:
         return self._call_ollama(prompt)
   
   def _call_openai(self, prompt: str) -> Dict[str, Any]:
      """Chiamata OpenAI con gestione errori"""
      response = self.client.chat.completions.create(
         model=self.model,
         messages=[
               {"role": "system", "content": "You are a PII classification expert. Output only valid JSON."},
               {"role": "user", "content": prompt}
         ],
         temperature=0.0,
         response_format={"type": "json_object"},
         max_tokens=150  # Limitiamo i token per risposte concise
      )
      return json.loads(response.choices[0].message.content)
   
   def _call_ollama(self, prompt: str) -> Dict[str, Any]:
      """Chiamata Ollama con gestione errori"""
      response = ollama.chat(
         model=self.model,
         messages=[{"role": "user", "content": prompt}],
         format="json",
         options={"temperature": 0.0, "num_predict": 150}
      )
      return json.loads(response["message"]["content"])
   
   def _validate_response(self, raw: Dict[str, Any], fallback_label: str) -> Dict[str, Any]:
      """Valida e normalizza la risposta LLM"""
      
      reasoning = raw.get("reasoning", "No reasoning provided")[:200]  # Limita lunghezza
      label = raw.get("corrected_label", fallback_label).strip().upper()
      
      # Validazione: se label non valida, usa fallback
      if label not in self.valid_labels:
         print(f"   [WARNING] Invalid label '{label}' → fallback to '{fallback_label}'")
         label = fallback_label
      
      return {
         "is_pii": label != "O",
         "corrected_label": label,
         "reasoning": reasoning
      }
   
   def _error_response(self, fallback_label: str, error_msg: str) -> Dict[str, Any]:
      """Response di fallback in caso di errore"""
      return {
         "is_pii": False,
         "corrected_label": fallback_label,
         "reasoning": f"Error: {error_msg}",
         "cached": False
      }
   
   def get_cache_stats(self) -> Optional[Dict[str, Any]]:
      """Ritorna statistiche cache"""
      return self.cache.get_stats() if self.cache else None


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================
if __name__ == "__main__":
   import os
   from dotenv import load_dotenv
   
   load_dotenv()
   
   # Inizializza router
   router = OptimizedLLMRouter(
      source="openai",
      api_key=os.getenv("OPENAI_API_KEY"),
      model="gpt-4o-mini",
      enable_cache=True,
      cache_size=500
   )
   
   # Test
   text = "My name is John Smith and I live in New York"
   result = router.disambiguate(
      target_token="John",
      full_text=text,
      char_start=11,
      char_end=15,
      current_pred="B-GIVENNAME",
      prev_label="O",
      lang="en"
   )
   
   print("\n=== RESULT ===")
   print(json.dumps(result, indent=2))
   
   # Statistiche cache
   if stats := router.get_cache_stats():
      print("\n=== CACHE STATS ===")
      print(json.dumps(stats, indent=2))