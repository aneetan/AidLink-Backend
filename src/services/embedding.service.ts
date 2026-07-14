import { DatasetProps, IntentProps } from "../types/embedding.types";

class EmbeddingService {
   private readonly dimension = 768;

   private hashToken(token: string): number {
      let hash = 5381;
      for (let i = 0; i < token.length; i++) {
         hash = ((hash << 5) + hash) + token.charCodeAt(i);
      }
      return hash;
   }

   private normalizeVector(vector: number[]): number[] {
      const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
      return norm === 0 ? vector : vector.map(value => value / norm);
   }

   // Generate a deterministic embedding vector for text.
   async generateEmbeddings(text: string): Promise<number[]> {
      const normalizedText = text
         .toLowerCase()
         .replace(/[\W_]+/g, ' ')
         .trim();

      const tokens = normalizedText.split(/\s+/).filter(Boolean);
      const vector = new Array(this.dimension).fill(0);

      tokens.forEach((token, tokenIndex) => {
         const hash = Math.abs(this.hashToken(token));
         const positionIndex = hash % this.dimension;
         vector[positionIndex] += 1 + (tokenIndex % 3) * 0.1;
         const secondaryIndex = (positionIndex + token.length) % this.dimension;
         vector[secondaryIndex] += 0.5;
      });

      return this.normalizeVector(vector);
   }

   // Generate embeddings for all intents in the dataset
   async generateAllEmbeddings(dataset: DatasetProps): Promise<IntentProps[]> {
      const embeddingIntents: IntentProps[] = [];

      for (const intent of dataset.intents) {
         const stepsText = intent.response.steps
            .map(step => `${step.step_number}. ${step.instruction}`)
            .join(' ');

         const fullText = [
            intent.intent_name,
            intent.response.context || '',
            stepsText,
            intent.response.additional_notes || '',
            `Warnings: ${intent.metadata.warnings.join(', ')}`
         ]
            .filter(Boolean)
            .join(' ')
            .trim();

         const embedding = await this.generateEmbeddings(fullText);

         embeddingIntents.push({
            ...intent,
            embeddings: embedding
         });

         await new Promise(resolve => setTimeout(resolve, 100));
      }

      return embeddingIntents;
   }
}

export default new EmbeddingService();