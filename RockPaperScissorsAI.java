package challenges;
import java.util.*;
import java.lang.Math;

public class RockPaperScissorsAI {
	private ArrayList<Integer> pastMoves;
	private final int NUM_INPUTS = 70;
	private final int NUM_HIDDEN_LEVELS = 10;
	private final int[] HIDDEN_LEVEL_SIZES;
	private final int NUM_OUTPUTS = 3;
	private final double ETA = 0.4;
	private final int NUM_REPS = 6;
	
	private double[][] biases; //first index is layer, second index is node
	private double[][][] weights; //first indicates from which layer to which (0 indicates from input to 1st hidden layer), second index is the to layer, third is the from layer
	private int[] network;
	private double[] answer;
	
	private double[][] a;
	private double[][] z;
	private double[][] error;
	
	{
		HIDDEN_LEVEL_SIZES = new int[]{100,100,100,100,100,100,100,100,100,100};
		pastMoves = new ArrayList<Integer>(NUM_INPUTS);
	}
	
	public RockPaperScissorsAI(){
		//generate a past array of moves
		Random rand = new Random();
		for (int i = 0; i < NUM_INPUTS; i++){
			pastMoves.add(rand.nextInt(3)+1);
		}
		
		//set up network size
		network = new int[NUM_HIDDEN_LEVELS+2];
		network[0]=NUM_INPUTS;
		for (int i = 0; i < NUM_HIDDEN_LEVELS; i++){
			network[i+1]=HIDDEN_LEVEL_SIZES[i];
		}
		network[NUM_HIDDEN_LEVELS+1]=NUM_OUTPUTS;
		
		//generate initial biases
		biases = new double[network.length-1][];
		for (int i = 0; i < network.length-1; i++){
			biases[i]=new double[network[i+1]];
			for (int j = 0; j < biases[i].length; j++){
				biases[i][j]=rand.nextDouble()*(rand.nextInt(2)-0.5)/0.5;
			}
		}
		
		//generate initial weights
		weights = new double[network.length-1][][];
		for (int i = 0; i < weights.length; i++){
			weights[i] = new double[network[i+1]][network[i]];
			for (int j = 0; j < weights[i].length; j++){
				for (int k = 0; k < weights[i][j].length; k++){
					weights[i][j][k]=rand.nextDouble()*(rand.nextInt(2)-0.5)/0.5;
				}
			}
		}
		
		//generate activation, weighted input and error arrays
		a = new double[network.length][];
		z = new double[network.length-1][];
		error = new double[network.length-1][];
		a[0] = new double[NUM_INPUTS];
		for (int i = 0; i < NUM_HIDDEN_LEVELS; i++){
			a[i+1] = new double[HIDDEN_LEVEL_SIZES[i]];
			z[i] = new double[HIDDEN_LEVEL_SIZES[i]];
			error[i] = new double[HIDDEN_LEVEL_SIZES[i]];
		}
		a[a.length-1] = new double[NUM_OUTPUTS];
		z[z.length-1] = new double[NUM_OUTPUTS];
		error[error.length-1] = new double[NUM_OUTPUTS];
	}
	
	//1 - rock, 2 - scissors, 3 - paper
	public void update(int move){
		//System.out.print("Predicted move: ");
		//printArray(answer);
		//System.out.print("Actual move: ");
		//printArray(convert(move));
		//move is desired outcome
		for (int r = 0; r < NUM_REPS; r++){
			//determine final error
			for (int i = 0; i < error[error.length-1].length; i++){
				error[error.length-1][i] = costDerivative(answer[i],convert(move)[i])*sigmoidPrime(z[error.length-1][i]);
			}
			
			//update weights and biases
			for (int i = network.length-1; i > 0; i--){ // i indicates which layer in the network is being processed
				//determine current layer error
				if (i < network.length-1){
					for (int j = 0; j < error[i-1].length; j++){ // j indicates the error node
						for (int k = 0; k < error[i].length; k++){ // k indicates the nodes on the next layer
							error[i-1][j] += weights[i][k][j]*error[i][k];
						}
						error[i-1][j] = error[i-1][j]*sigmoidPrime(z[i-1][j]);
					}
				}
				//update weights
				for (int j = 0; j < weights[i-1].length; j++){ // j indicates which node in the current layer is being processed
					for (int k = 0; k < weights[i-1][j].length; k++){ // k indicates the node in the previous layer onto which the weight applies
						weights[i-1][j][k] = weights[i-1][j][k] - ETA*a[i-1][k]*error[i-1][j];
					}
				}
				//update biases
				for (int j = 0; j < biases[i-1].length; j++){
					biases[i-1][j] = biases[i-1][j] - ETA*error[i-1][j];
				}
			}
		}
		
		//update moves
		pastMoves.remove(0);
		pastMoves.add(move);
	}
	
	public double[] convert(int move){
		double[] answer = new double[NUM_OUTPUTS];
		for (int i = 0; i < answer.length; i++){
			answer[i]=0.0;
		}
		answer[move-1]=1;
		return answer;
	}
	
	public int next(){
		//convert pastMoves into a double array
		double[] fromNodes = new double[pastMoves.size()];
		for (int i = 0; i < fromNodes.length; i++){
			fromNodes[i]=pastMoves.get(i);
			a[0][i] = fromNodes[i];
		}
		
		double[] toNodes = new double[1];
		for (int i = 0; i < weights.length; i++){ // i indicates which 2 layers are interacting, i=0 means layers 0 and 1
			toNodes = new double[network[i+1]];
			for (int j = 0; j < toNodes.length; j++){ //j indicates the node on the to layer being processed
				z[i][j] = dot(weights[i][j],fromNodes)+biases[i][j];
				a[i+1][j] = sigmoid(z[i][j]);
				toNodes[j]=a[i+1][j];
			}
			fromNodes = toNodes.clone();
		}
		answer = toNodes;
		return interpret(toNodes);
	}
	
	//interprets an output by the AI
	public int interpret(double[] a){
		if (a[0]>a[1]&&a[0]>a[2]){
			return 2;
		}else if (a[1]>a[2]){
			return 3;
		}else{
			return 1;
		}
	}
	
	public double[][] getBiases(){
		return biases;
	}
	
	public double dot(double[] a, double[]b){
		return 1.0;
	}
	
	public double costDerivative(double output, double actual){
		return output-actual;
	}
	
	public double sigmoid(double z){
		return 1.0/(1.0+Math.exp(-z));
	}
	
	public void sigmoid(double[] z){
		for (int i = 0; i < z.length; i++){
			z[i]=sigmoid(z[i]);
		}
	}
	
	public double sigmoidPrime(double z){
		return sigmoid(z)*(1-sigmoid(z));
	}
	
	public void sigmoidPrime(double[] z){
		for (int i = 0; i < z.length; i++){
			z[i]=sigmoidPrime(z[i]);
		}
	}
	
	public void printArray(double[] arr){
		for (int i = 0; i < arr.length; i++){
			System.out.print(arr[i]+" ");
		}
		System.out.println();
	}
}
