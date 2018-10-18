int paddle_beam_decode(torch::Tensor th_probs,
                       torch::Tensor th_seq_lens,
                       const char* labels,
                       int vocab_size,
                       size_t beam_size,
                       size_t num_processes,
                       double cutoff_prob,
                       size_t cutoff_top_n,
                       size_t blank_id,
                       torch::Tensor th_output,
                       torch::Tensor th_timesteps,
                       torch::Tensor th_scores,
                       torch::Tensor th_out_length);

int paddle_beam_decode_lm(torch::Tensor th_probs,
                          torch::Tensor th_seq_lens,
                          const char* labels,
                          int vocab_size,
                          size_t beam_size,
                          size_t num_processes,
                          double cutoff_prob,
                          size_t cutoff_top_n,
                          size_t blank_id,
                          void *scorer,
                          torch::Tensor th_output,
                          torch::Tensor th_timesteps,
                          torch::Tensor th_scores,
                          torch::Tensor th_out_length);

void* paddle_get_scorer(double alpha,
                        double beta,
                        const char* lm_path,
                        const char* labels,
                        int vocab_size);

int is_character_based(void *scorer);
size_t get_max_order(void *scorer);
size_t get_dict_size(void *scorer);
void reset_params(void *scorer, double alpha, double beta);
