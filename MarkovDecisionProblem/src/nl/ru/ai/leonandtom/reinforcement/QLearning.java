package nl.ru.ai.leonandtom.reinforcement;

import java.util.Formatter;
import java.util.HashMap;
import java.util.Random;

import nl.ru.ai.vroon.mdp.*;

/**
 * class that can run the QLearning algorithm
 * 
 * @author Leon Driessen s4791835 & Tom Kamp s4760921
 *
 */

public class QLearning
{

  private MarkovDecisionProblem mdp;
  private final double DISCOUNT, LEARNINGRATE, PENALTY;
  private double epsilon;
  private int epochs, width, height;
  private HashMap<Action, Double>[][] QValues;
  private Random random;

  /**
   * constructor for the QLearning class
   * 
   * @param MarkovDecisionProblem mdp
   * @param Double                discount
   * @param Double                epsilon
   * @param Double                learningRate
   * @param Double                penalty
   * @param Int                   epochs
   */
  public QLearning(MarkovDecisionProblem mdp, double discount, double epsilon, double learningRate, double penalty, int epochs)
  {
    this.mdp=mdp;
    this.DISCOUNT=discount;
    this.epsilon=epsilon;
    this.LEARNINGRATE=learningRate;
    this.PENALTY=penalty;
    this.epochs=epochs;
    this.width=mdp.getWidth();
    this.height=mdp.getHeight();
    this.QValues=zeroInitialization();
    this.random=new Random();
  }

  /**
   * function to let the algorithm run
   */
  public void run()
  {

    // loop through the given amount of epochs
    for(int epoch=0;epoch<epochs;epoch++)
    {

      // restart the mdp and set a random state as the start state
      mdp.restart();
      setStartState();

      // loop until the mdp has reached a terminal state
      while(!mdp.isTerminated())
      {
        int row=mdp.getStateXPosition();
        int col=mdp.getStateYPostion();
        Action action=null;

        // Search for an action using greedy search
        if(Math.random()>epsilon)
        {
          HashMap<Action, Double> qValue=QValues[row][col];
          action=getBestAction(qValue);
        }
        else
        {
          Action[] actions=Action.values();
          action=actions[random.nextInt(Action.values().length)];
        }

        // perform the action and see what the Qvalue is
        double reward=mdp.performAction(action);
        double curQValue=QValues[row][col].get(action);

        // look for the best action in the next state and calculate the qvalue using bellman's
        int newrow=mdp.getStateXPosition();
        int newcol=mdp.getStateYPostion();
        Action bestAction=getBestAction(QValues[newrow][newcol]);
        double bestActionValue=QValues[newrow][newcol].get(bestAction);
        double newQValue=curQValue+LEARNINGRATE*(reward-PENALTY+DISCOUNT*bestActionValue-curQValue);
        QValues[row][col].put(action,newQValue);
      }
      //reduce the randomness
      epsilon = epsilon*0.95;
    }

    // print the policy and values
    printPolicyAndValues();
  }

  /**
   * loops through all actions and finds the action with the highest value
   * 
   * @param HashMap<Action, Double> QValue
   * @return Action bestAction
   */
  private Action getBestAction(HashMap<Action, Double> QValue)
  {

    // initialize with a value lower than the lowest reward
    double maxValue=mdp.getNegReward()-1;
    Action bestAction=null;

    // loop through all actions and save the action with the maximum value
    for(Action action : QValue.keySet())
    {
      Double curValue=QValue.get(action);
      if(curValue>maxValue)
      {
        maxValue=curValue;
        bestAction=action;
      }
    }
    return bestAction;
  }

  /**
   * sets the start state to a random state calls itself if the chosen state is an
   * end state.
   */
  private void setStartState()
  {

    // pick a random field and set this as its initial state
    int row=random.nextInt(width);
    int col=random.nextInt(height);
    mdp.setInitialState(row,col);
    Field field=mdp.getField(row,col);

    // if that field is an end state, call the function again until the initial
    // state is not an end state
    if(field==Field.REWARD||field==Field.NEGREWARD)
    {
      setStartState();
    }

  }

  /**
   * Initializes the QValues as zero's
   * 
   * @return HashMap<Action, Double> QValues
   */
  @SuppressWarnings("unchecked")
  private HashMap<Action, Double>[][] zeroInitialization()
  {

    // loop through all fields and put the actions with the probabilities together
    // in pairs
    HashMap<Action, Double>[][] QValues=new HashMap[width][height];
    for(int row=0;row<width;row++)
    {
      for(int col=0;col<height;col++)
      {
        QValues[row][col]=new HashMap<Action, Double>();
        for(Action action : Action.values())
        {
          QValues[row][col].put(action,0.0);
        }
      }
    }
    return QValues;
  }

  /**
   * prints the policy and the values
   */
  private void printPolicyAndValues()
  {

    // create stringbuilders and pass these to formatters to make tables using
    // format code
    StringBuilder policy=new StringBuilder();
    StringBuilder values=new StringBuilder();
    Formatter policyFormatter=new Formatter(policy);
    Formatter valuesFormatter=new Formatter(values);

    // loop through all fields from back to front
    for(int col=height-1;col>=0;col--)
    {
      policy.append("| ");
      values.append("| ");
      for(int row=0;row<width;row++)
      {
        Field field=mdp.getField(row,col);

        // check the field to see what if you need to print an action/value or
        // that its a neg end, end or obstacle ()
        switch(field)
        {
          case EMPTY:
            Action bestAction=getBestAction(QValues[row][col]);
            policyFormatter.format("%-6s",bestAction);
            valuesFormatter.format("%-6s",Math.round(QValues[row][col].get(bestAction)*100.0)/100.0);
            break;
          case NEGREWARD:
            policyFormatter.format("%-6s","NEG");
            valuesFormatter.format("%-6s","NEG");
            break;
          case OBSTACLE:
            policyFormatter.format("%-6s","()");
            valuesFormatter.format("%-6s","()");
            break;
          case REWARD:
            policyFormatter.format("%-6s","END");
            valuesFormatter.format("%-6s","END");
            break;
          case OUTOFBOUNDS:
            policyFormatter.format("%-6s","OOB");
            valuesFormatter.format("%-6s","OOB");
        }
        policy.append(" | ");
        values.append(" | ");

      }
      policy.append("\n");
      values.append("\n");
    }
    // close the formatters to prevent memory leaks
    policyFormatter.close();
    valuesFormatter.close();

    // prints the stringbuilers
    System.out.println("Policy: \n"+policy.toString()+"\n");
    System.out.println("Values: \n"+values.toString()+"\n");
  }

}
