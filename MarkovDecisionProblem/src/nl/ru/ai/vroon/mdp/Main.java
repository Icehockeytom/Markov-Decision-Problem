package nl.ru.ai.vroon.mdp;

import nl.ru.ai.leonandtom.reinforcement.*;

/**
 * This main is for testing purposes (and to show you how to use the MDP class).
 * 
 * @author Jered Vroon & Leon Driessen s4791835 & Tom Kamp s4760921
 *
 */
public class Main
{

  /**
   * @param args, not used
   */
  public static void main(String[] args)
  {
    		MarkovDecisionProblem mdp = new MarkovDecisionProblem();
    		mdp.setInitialState(0, 0);
    		mdp.setShowProgress(false);
    		for (int i = 0; i < 100; i++){
    			mdp.performAction(Action.UP);
    			mdp.performAction(Action.RIGHT);
    			mdp.performAction(Action.DOWN);
    			mdp.performAction(Action.LEFT);
    			mdp.restart();
    		}

//    MarkovDecisionProblem mdp2=new MarkovDecisionProblem(10,10);
//    mdp2.setField(7,7,Field.REWARD);
//    mdp2.setField(4,4,Field.OBSTACLE);
//    mdp2.setShowProgress(false);
//    for(int i=0;i<100;i++)
//    {
//      mdp2.performAction(Action.UP);
//      mdp2.performAction(Action.RIGHT);
//      mdp2.performAction(Action.DOWN);
//      mdp2.performAction(Action.LEFT);
//    }
    

//    MarkovDecisionProblem mdp3=new MarkovDecisionProblem(10,10);
//    mdp3.setField(7,7,Field.REWARD);
//    mdp3.setField(4,4,Field.NEGREWARD);
//    mdp3.setShowProgress(false);
//    for(int i=0;i<100;i++)
//    {
//      mdp3.performAction(Action.UP);
//      mdp3.performAction(Action.RIGHT);
//      mdp3.performAction(Action.DOWN);
//      mdp3.performAction(Action.LEFT);
//    }

    		ValueIteration vi = new ValueIteration(mdp, 0.5, 0.1);
    		vi.run();
//    QLearning ql=new QLearning(mdp,0.5,0.02,0.2,0.0,1000);
//    ql.run();
  }
}
