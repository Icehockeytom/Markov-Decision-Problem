package nl.ru.ai.leonandtom.reinforcement;

import java.util.Formatter;

import nl.ru.ai.vroon.mdp.*;

/**
 * class that can run the value iteration algorithm
 * 
 * @author Leon Driessen s4791835 & Tom Kamp s4760921
 *
 */
public class ValueIteration
{

  private MarkovDecisionProblem mdp;
  private final double DISCOUNT, POSREWARD, NEGREWARD, NOREWARD, PENALTY;
  private final double DELTA=1E-6;
  private boolean isDeterministic=true;
  private int width, height, counter;
  private double[][] currentStates, nextStates;
  private double[] transitionProbabilities;
  private Action[][] policy;

  /**
   * constructor for the ValueIteration class
   * 
   * @param Markov Decision Problem mdp
   * @param Double gamma : discount factor
   * @param Double delta
   * @param Double penalty
   */
  public ValueIteration(MarkovDecisionProblem mdp, double discount, double penalty)
  {
    this.mdp=mdp;
    this.DISCOUNT=discount;
    this.PENALTY=penalty;
    this.POSREWARD=mdp.getPosReward();
    this.NEGREWARD=mdp.getNegReward();
    this.NOREWARD=mdp.getNoReward();
    this.isDeterministic=mdp.isDeterministic();
    this.width=mdp.getWidth();
    this.height=mdp.getHeight();
    this.currentStates=zeroInitialization();
    this.nextStates=zeroInitialization();
    this.transitionProbabilities=mdp.getTransitionProbabilities();
    this.policy=new Action[width][height];
  }

  /**
   * To let the algorithm run
   */
  public void run()
  {

    // runs until converged,
    // this becomes true when the last checked states is the same as the next state
    boolean hasConverged=false;
    while(counter++>=0&&!hasConverged)
    {

      // loop through all fields
      for(int row=0;row<width;row++)
      {
        for(int col=0;col<height;col++)
        {
          Field field=mdp.getField(row,col);

          // check if the field terminates or not
          if(field==Field.REWARD||field==Field.NEGREWARD)
          {
            continue;
          }

          // calculates the max qvalue by iterating through all qvalues
          double maxValue=NEGREWARD-1;
          for(Action action : Action.values())
          {
            double Qvalue=calculateQvalues(row,col,action);
            if(Qvalue>maxValue)
            {
              maxValue=Qvalue;
            }
          }
          nextStates[row][col]=maxValue;
          hasConverged=Math.abs(currentStates[row][col]-nextStates[row][col])<DELTA;
        }
      }
      // move to the next states
      currentStates=nextStates;
      nextStates=new double[width][height];
    }
    System.out.println("The amount of iterations until convergence: "+counter);

    // calculates the best policies and prints those + the values of the states
    getPolicy();
    printPolicyAndValues();
  }

  /**
   * Calculates the q values for a state and an action
   * 
   * @param Int    row
   * @param Int    col
   * @param Action action
   * @return Double q value
   */
  private double calculateQvalues(int row, int col, Action action)
  {
    double sum=0;
    double[] transitionProbability=getTransitionProbabilities();
    Action[] actions=new Action[] { action, Action.nextAction(action), Action.previousAction(action), Action.backAction(action) };

    // checks all actions
    for(int i=0;i<actions.length;i++)
    {

      // checks the directional difference from all actions
      // adds them to the row and col to get the new rows
      int[] actiontransition=getActionTransition(actions[i]);
      int addrow=actiontransition[0];
      int addcol=actiontransition[1];
      int newrow=row+addrow;
      int newcol=col+addcol;

      // checks if the action is legal
      // sums the value using the bellman's formula
      if(illegalAction(row,col,actions[i]))
      {
        Field field=mdp.getField(row,col);
        sum+=transitionProbability[i]*(getReward(field)-PENALTY+DISCOUNT*currentStates[row][col]);
      }
      else
      {
        Field field=mdp.getField(newrow,newcol);
        sum+=transitionProbability[i]*(getReward(field)-PENALTY+DISCOUNT*currentStates[newrow][newcol]);
      }
    }
    return sum;
  }

  /**
   * Calculates the best action given a state
   * 
   * @param int row
   * @param int col
   * @return Action bestAction
   */
  private Action calculateBestAction(int row, int col)
  {

    // sets the maxvalue as something lower than the lowest possible value
    double maxValue=NEGREWARD-1;
    Action bestAction=null;
    double curValue=0;

    // loops through all actiosn
    for(int i=0;i<Action.values().length;i++)
    {

      // if the action is not allowed we stay in the same field
      if(illegalAction(row,col,Action.values()[i]))
      {
        curValue=currentStates[row][col];
        if(curValue>maxValue)
        {
          maxValue=curValue;
          bestAction=Action.values()[i];
        }
      }
      else
      {

        // checks the directional difference from all actions
        // adds them to the row and col to get the new rows
        int[] actiontransition=getActionTransition(Action.values()[i]);
        int addrow=actiontransition[0];
        int addcol=actiontransition[1];
        int newrow=row+addrow;
        int newcol=col+addcol;
        Field field=mdp.getField(newrow,newcol);

        // checks the reward gained from doing this action
        curValue=getReward(field)-PENALTY+DISCOUNT*currentStates[newrow][newcol];
        if(curValue>maxValue)
        {
          maxValue=curValue;
          bestAction=Action.values()[i];
        }
      }
    }
    return bestAction;
  }

  /**
   * Checks if a given action is legal in a given state
   * 
   * @param        int row
   * @param        int col
   * @param Action action
   * @return boolean
   */
  private boolean illegalAction(int row, int col, Action action)
  {
    int[] actiontransition=getActionTransition(action);
    int addrow=actiontransition[0];
    int addcol=actiontransition[1];
    // checks if the next move makes you go outofbounds
    // when we used the outofbounds property of the field it did not work so we
    // manually check
    return (row+addrow<0||row+addrow>=width)||(col+addcol<0||col+addcol>=height)||(mdp.getField(row+addrow,col+addcol)==Field.OBSTACLE);
  }

  /**
   * fills the policy array with the best action for every state
   */
  private void getPolicy()
  {

    // loop through all fields
    for(int row=0;row<width;row++)
    {
      for(int col=0;col<height;col++)
      {
        Field field=mdp.getField(row,col);

        // check if that field is an end state
        if(!(field==Field.REWARD||field==Field.NEGREWARD))
        {
          policy[row][col]=calculateBestAction(row,col);

        }
      }
    }
  }

  /**
   * Gets the transition probability array
   * 
   * @return double[] filled with probabilities
   */
  private double[] getTransitionProbabilities()
  {
    double stepProbability=transitionProbabilities[0];
    double sideStepProbability=transitionProbabilities[1];
    double backStepProbability=transitionProbabilities[2];

    // if the mdp is deterministic we always know what action we are going to take
    // so our
    // transition prob is 100% for the step
    if(isDeterministic)
    {
      return new double[] { 1, 0, 0, 0 };
    }
    else
    {

      // else we just return the probability for each possible step
      return new double[] { stepProbability, sideStepProbability/2, sideStepProbability/2, backStepProbability };
    }
  }

  /**
   * initializes double[][] as an array filled with zeros
   * 
   * @return Double[][] zero
   */
  private double[][] zeroInitialization()
  {
    double[][] zeros=new double[width][height];
    for(int row=0;row<width;row++)
    {
      for(int col=0;col<height;col++)
      {
        zeros[row][col]=0;
      }
    }
    return zeros;
  }

  /**
   * gets the transition movement in a 2d space given a move.
   * 
   * @param Action action
   * @return int[]
   */
  private int[] getActionTransition(Action action)
  {
    switch(action)
    {
      case UP:
        return new int[] { 0, 1 };
      case DOWN:
        return new int[] { 0, -1 };
      case LEFT:
        return new int[] { -1, 0 };
      case RIGHT:
        return new int[] { 1, 0 };
      default:
        return null;

    }
  }

  /**
   * gets the reward given a state
   * 
   * @param Field field
   * @return double
   */
  private double getReward(Field field)
  {
    switch(field)
    {
      case REWARD:
        return POSREWARD;
      case NEGREWARD:
        return NEGREWARD;
      case EMPTY:
        return NOREWARD;
      default:
        return 0;
    }
  }

  /**
   * Prints the policies and the values
   */
  public void printPolicyAndValues()
  {

    // create stringbuilders and pass these to formatters to make tables using
    // format code
    StringBuilder policysb=new StringBuilder();
    StringBuilder values=new StringBuilder();
    Formatter policyFormatter=new Formatter(policysb);
    Formatter valuesFormatter=new Formatter(values);

    // loop through all fields from back to front
    for(int col=height-1;col>=0;col--)
    {
      policysb.append("| ");
      values.append("| ");
      for(int row=0;row<width;row++)
      {
        Field field=mdp.getField(row,col);

        // check the field to see what if you need to print a policy/value or
        // that its a neg end, end or obstacle ()
        switch(field)
        {
          case EMPTY:
            policyFormatter.format("%-6s",policy[row][col]);
            valuesFormatter.format("%-6s",Math.round(currentStates[row][col]*100.0)/100.0);
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
        policysb.append(" | ");
        values.append(" | ");

      }
      policysb.append("\n");
      values.append("\n");
    }

    // close the formatters to prevent memory leaks
    policyFormatter.close();
    valuesFormatter.close();

    // prints the stringbuilers
    System.out.println("Policy: \n"+policysb.toString()+"\n");
    System.out.println("Values: \n"+values.toString()+"\n");
  }

}
