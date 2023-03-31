import openai
import os
import time

class GPTAnswer:
    def __init__(self, modelname="gpt-3.5-turbo"):
        openai.organization = os.getenv("OPENAI_ORG_ID")
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.modelname = modelname

    def generate_answer(self, premise_questions):
        answers = []
        if "gpt-3.5" in self.modelname:
            for premise_question in premise_questions:
                response = openai.ChatCompletion.create(
                                model=self.modelname,
                                messages=[
                                    {"role": "user", "content": premise_question}
                                    ],
                                max_tokens=512,
                                n=1,
                                stop=None,
                                temperature=0,
                            )
            answers.append(response["choices"][0].message.content)
        else:
            completion = openai.Completion.create(
                            engine=self.modelname,
                            prompts=premise_questions,
                            max_tokens=512,
                            n=1,
                            stop=None,
                            temperature=0,
                        )
            for choice in completion.choices:
                answers.append(choice.text)
        return answers

    def log_answer(self, qtype, premise_questions={}, output_path=""):
        question_ids = list(premise_questions.keys())
        premise_questions = list(premise_questions.values())
        predicted_answers = self.generate_answer(premise_questions)
        time.sleep(60)
        for idx, prediction in enumerate(predicted_answers):
            with open(os.path.join(f"{output_path}",f"{qtype}_{question_ids[idx]}_predicted_answer.txt"),"w") as prediction_file:
                print(prediction, file=prediction_file)