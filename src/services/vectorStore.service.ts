import { FindSimilarVectorsResult, IntentProps, PineconeMatch, VectorMetadata, VectorProps } from "../types/embedding.types";
import PineconeConfig from "../config/pinecone.config";
require('dotenv').config();

class VectorStoreService {
   private pineconeConfig: PineconeConfig;
   private indexName: string;
   private vectorDimensions: number;

   constructor() {
      this.pineconeConfig = PineconeConfig.getInstance();
      this.indexName = 'firstaid-embeddings';
      this.vectorDimensions = 768;
   }

   //initialize service
   async initialize(): Promise<void> {
      try {
         await this.pineconeConfig.initialize();
      } catch (error) {
         console.error('Failed to initialize VectorStoreService:', error);
         throw error;
      }
   }

    // Get the index instance
   private getIndex() {
      return this.pineconeConfig.getIndex(this.indexName);
   }


   // Add embedded intents to vector store
   async addVectors (embeddedIntents: IntentProps[]): Promise<void> {
      try {
         const index = this.getIndex();
         const stats = await index.describeIndexStats();
         if (stats.totalRecordCount && stats.totalRecordCount > 0) {
            console.log('Vectors already exist in Pinecone. Skipping addition.');
            return;
         }

          const validIntents = embeddedIntents.filter(intent => {
            const isValid = intent.embeddings && intent.embeddings.length === this.vectorDimensions;
            if (!isValid) {
               console.warn(`Skipping intent "${intent.intent_name}" - invalid embedding dimension: ${intent.embeddings?.length || 0}`);
            }
            return isValid;
         });

         if (validIntents.length === 0) {
            throw new Error('No valid embeddings found. All embeddings have dimension 0.');
         }

         const vectors: VectorProps[] = embeddedIntents.map((intent, idx) => ({
            id: `intent-${idx}-${intent.intent_name.replace(/\s+/g, '-').toLowerCase()}`,
            values: intent.embeddings ?? [],
            metadata : {
               intent_name: intent.intent_name,
               immediate_action: intent.response.immediate_action || '',
               context: intent.response.context || '',
               steps: JSON.stringify(intent.response.steps),
               additional_notes: intent.response.additional_notes || '',
               warnings: JSON.stringify(intent.metadata.warnings),
               source: intent.metadata.source,
               last_updated: intent.metadata.last_updated,
               user_queries: JSON.stringify(intent.user_queries)
            } as VectorMetadata
         }));

         // Upsert vectors in batches (Pinecone recommends batches of 100)
         const batchSize = 100;

         for (let i = 0; i < vectors.length; i += batchSize) {
            const batch = vectors.slice(i, i + batchSize);
            await index.upsert(batch);
         }

      } catch (e) {
         console.error('Error adding vectors to Pinecone:', e);
         throw e;
      }
   }

    // Find the most similar intents to a query
   async findSimilarVectors(
      queryEmbedding: number[],
      topK: number = 3
   ): Promise<FindSimilarVectorsResult[]> {
      try {
         const index = this.getIndex();
         const queryResponse = await index.query({
            vector: queryEmbedding,
            topK: topK,
            includeMetadata: true,
            includeValues: false
         });

         // Transform Pinecone results to match our expected format
         return queryResponse.matches
            .filter((match): match is PineconeMatch & { score: number; metadata: Record<string, any> } => 
               match.score !== undefined && 
               match.score >= 0.4 && 
               match.metadata !== undefined
            )
            .map(match => ({
               similarity: match.score,
               intent: {
                  intent_name: match.metadata.intent_name,
                  response: {
                     immediate_action: match.metadata.immediate_action,
                     context: match.metadata.context,
                     steps: JSON.parse(match.metadata.steps),
                     additional_notes: match.metadata.additional_notes
                  },
                  metadata: {
                     warnings: JSON.parse(match.metadata.warnings),
                     source: match.metadata.source,
                     last_updated: match.metadata.last_updated
                  },
                  user_queries: JSON.parse(match.metadata.user_queries)
               }
            }));
      } catch (e) {
         console.error('Error querying Pinecone:', e);
         throw e;
      }
   }
}

export default new VectorStoreService();