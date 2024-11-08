from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
import json



class Course:
  def __init__(self, materials):
    
    self.materials=json.load(open(materials))
    
    self.system_prompt_teacher = '''
    Review and evaluate a student-created study strategy in relation to specific course material, providing constructive feedback to improve learning effectiveness.
    
    Ensure the feedback highlights areas for enhancement, reinforces strengths, and includes actionable recommendations that align with the course content.
    
    Steps to Complete the Task
    Review the Course Material:
    Understand the main themes, key concepts, and critical points in the provided material.
    Analyze the Student's Strategy:
    Identify the goals and methods described by the student.
    Pinpoint any strengths or effective techniques already present in the strategy.
    Evaluate Limitations:
    Detect potential gaps, inefficiencies, or weaknesses in the approach.
    Develop Feedback:
    Formulate specific, constructive suggestions for improvement.
    Recommend techniques or changes that align directly with the course material to enhance learning retention and comprehension.
    Output Format
    Summary: A brief overview of the strengths of the current strategy.
    Detailed Feedback: Constructive points addressing areas for improvement, with actionable advice.
    Recommendations: 2-3 specific study methods or changes tailored to the course content.
    Conclusion: A motivational note encouraging the student to apply the feedback and adjust as needed.
    '''
    
    self.system_prompt_student = '''
    
    Develop a comprehensive and effective learning guideline to maximize the absorption of course content. Use provided course materials and teacher guidance to refine your strategy iteratively, aiming for continuous improvement and better exam performance.
    
    Understand the course objectives, key concepts, and skills to be acquired.
    Analyze the structure and content of the course material, identifying primary topics and subtopics.
    Create a study plan with a balanced schedule that includes review sessions, active learning, and practice exercises.
    Apply active recall and spaced repetition techniques for better retention.
    Practice with sample questions, previous exams, or real-world applications to test understanding.
    Incorporate teacher feedback to adjust strategies, focusing on weaker areas or overlooked content.
    Steps
    Analyze Course Material:
    
    Identify key concepts and themes.
    Break down complex topics into smaller, manageable parts.
    Create a Study Plan:
    
    Allocate study sessions with breaks for review.
    Schedule periodic self-assessments.
    Apply Learning Techniques:
    
    Use active recall by summarizing notes from memory.
    Implement spaced repetition for long-term retention.
    Self-Testing:
    
    Practice with mock exams or application-based questions.
    Reflect on mistakes and adjust focus areas.
    Refine Based on Feedback:
    
    Adapt study strategies based on teacher input.
    Reassess and incorporate improvements in real time.
    Output Format
    The output should be a well-structured study guideline document. Use bullet points for clarity.
    Highlight key techniques and time allocations.
    Include sections for initial strategy and periodic adjustments.
    Use subheadings such as Initial Study Plan, Self-Testing Methods, Feedback Integration, etc.
    '''

    

    teacher_template = """
    System: {system_prompt}
    
    course material: {material}
    
    student's learning strategy: {strategy}
    
    Your advice to improve the strategy:"""

    
    student_template = """
    System: {system_prompt}
    
    course material: {material}
    
    The last learning guideline is:{strategy}
    
    teacher's advice for the last guideline: {advice}
    
    The improved guideline:"""

    self.student_prompt = ChatPromptTemplate.from_template(student_template)
    self.teacher_prompt = ChatPromptTemplate.from_template(teacher_template)
  
    

    def TakeCourse(self, teacher, student, week):
      material=self.material[week*7:(week+1)*7]

      teacher_chain = self.teacher_prompt | teacher
      student_chain = self.student_prompt | student
      
      teacher_advice='None.'
      student_strategy='None.'

      # Process each three-page chunk
      for i in range(5):
      
          student_strategy = student_chain.invoke({'system_prompt':self.system_prompt_student, 'material':material, 'advice': teacher_advice, 'strategy':student_strategy})
      
          print(f'Student: {student_strategy}')
      
          print('////////////////////////////////////////////////////////////')
          print()
      
          teacher_advice = teacher_chain.invoke({'system_prompt':self.system_prompt_teacher, 'material':material, 'strategy':student_strategy})
      
          print(f'Teacher: {teacher_advice}')
          print('////////////////////////////////////////////////////////////')
          print()
      
      student_strategy = student_chain.invoke({'system_prompt':system_prompt_student, 'material':material, 'advice': teacher_advice, 'strategy':student_strategy})
      
      print(f'Student: {student_strategy}')

      return student_strategy

      
      

    
    
    
